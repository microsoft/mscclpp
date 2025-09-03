// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <ctime>
#include <numeric>

using json = nlohmann::json;

namespace mscclpp {

std::string VariableContext::substituteVariables(const std::string& template_str) const {
  std::string result = template_str;
  
  // Substitute ${VARIABLE_NAME} patterns
  std::regex var_pattern(R"(\$\{([^}]+)\})");
  std::smatch match;
  
  while (std::regex_search(result, match, var_pattern)) {
    std::string var_name = match[1].str();
    auto it = variables.find(var_name);
    if (it != variables.end()) {
      result.replace(match.position(), match.length(), it->second);
    } else {
      // Leave unresolved variables as-is for now
      break;
    }
  }
  
  return result;
}

// Fix member initialization order: rank_ should be initialized before isDynamic_
DynamicExecutionPlan::DynamicExecutionPlan(const std::string& planPath, int rank)
    : rank_(rank), name_(""), collective_(""), protocol_(""), isDynamic_(false), 
      minMessageSize_(0), maxMessageSize_(0), numThreadsPerBlock_(1024) {
  loadFromJson(planPath);
}

void DynamicExecutionPlan::loadFromJson(const std::string& planPath) {
  std::ifstream file(planPath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open dynamic execution plan file: " + planPath);
  }
  
  json j;
  file >> j;
  
  // Parse basic plan information
  name_ = j.value("name", "dynamic_plan");
  collective_ = j.value("collective", "alltoallv");
  protocol_ = j.value("protocol", "dynamic");
  isDynamic_ = j.value("dynamic", true);
  minMessageSize_ = j.value("min_message_size", 0);
  maxMessageSize_ = j.value("max_message_size", 1048576);
  numThreadsPerBlock_ = j.value("num_threads_per_block", 1024);
  
  // Parse dynamic parameters
  if (j.contains("dynamic_parameters")) {
    for (auto& [key, value] : j["dynamic_parameters"].items()) {
      dynamicParams_[key] = value.get<std::string>();
    }
  }
  
  // Parse GPU templates
  if (j.contains("gpus")) {
    for (auto& gpu_json : j["gpus"]) {
      DynamicGpuTemplate gpu_template;
      gpu_template.id = gpu_json.value("id", 0);
      gpu_template.inputChunks = gpu_json.value("input_chunks", "${DYNAMIC_INPUT_CHUNKS}");
      gpu_template.outputChunks = gpu_json.value("output_chunks", "${DYNAMIC_OUTPUT_CHUNKS}");
      gpu_template.scratchChunks = gpu_json.value("scratch_chunks", 0);
      
      // Parse operation templates
      if (gpu_json.contains("operations")) {
        for (auto& op_json : gpu_json["operations"]) {
          if (op_json.contains("operation_template")) {
            DynamicOperationTemplate op_template;
            auto& op_tmpl = op_json["operation_template"];
            op_template.type = op_tmpl.value("type", "put");
            op_template.inputChunk = op_tmpl.value("inputChunk", "${chunk_id}");
            op_template.outputChunk = op_tmpl.value("outputChunk", "${chunk_id}");
            op_template.peer = op_tmpl.value("peer", "${peer_rank}");
            op_template.channel = op_tmpl.value("channel", "0");
            op_template.threadblockCount = op_tmpl.value("threadblock_count", "${tb_count}");
            op_template.size = op_tmpl.value("size", "${chunk_size}");
            op_template.step = op_tmpl.value("step", "${step_id}");
            
            gpu_template.operationTemplates.push_back(op_template);
          }
        }
      }
      
      gpuTemplates_.push_back(gpu_template);
    }
  }
}

int DynamicExecutionPlan::calculateThreadBlocks(size_t messageSize) const {
  auto it = dynamicParams_.find("max_thread_blocks");
  int maxThreadBlocks = it != dynamicParams_.end() ? std::stoi(it->second) : 32;
  
  it = dynamicParams_.find("block_size");
  size_t blockSize = it != dynamicParams_.end() ? std::stoull(it->second) : 32768;
  
  int neededBlocks = (messageSize + blockSize - 1) / blockSize;
  return std::min(neededBlocks, maxThreadBlocks);
}

std::string DynamicExecutionPlan::instantiate(const DynamicRuntimeParams& params) {
  // Generate concrete JSON from template
  json concrete_json;
  
  // Basic plan information
  concrete_json["name"] = name_ + "_instantiated";
  concrete_json["collective"] = collective_;
  concrete_json["protocol"] = protocol_;
  concrete_json["inplace"] = false;
  concrete_json["num_threads_per_block"] = numThreadsPerBlock_;
  concrete_json["min_message_size"] = minMessageSize_;
  concrete_json["max_message_size"] = maxMessageSize_;
  
  // Generate concrete GPU information
  json gpus_json = json::array();
  
  for (const auto& gpu_template : gpuTemplates_) {
    if (gpu_template.id == rank_) {  // Only process our GPU
      json gpu_json;
      gpu_json["id"] = gpu_template.id;
      gpu_json["input_chunks"] = params.peerRanks.size();
      gpu_json["output_chunks"] = params.peerRanks.size();
      gpu_json["scratch_chunks"] = gpu_template.scratchChunks;
      gpu_json["channels"] = json::array();
      gpu_json["nvls_channels"] = json::array();
      
      // Generate concrete operations
      json operations = json::array();
      
      for (size_t peer_idx = 0; peer_idx < params.peerRanks.size(); ++peer_idx) {
        int peer_rank = params.peerRanks[peer_idx];
        size_t send_size = params.sendSizes[peer_idx];
        size_t recv_size = params.recvSizes[peer_idx];
        
        if (send_size > 0) {  // Generate send operation
          int tb_count = calculateThreadBlocks(send_size);
          
          json send_op;
          send_op["type"] = "put";
          send_op["inputChunk"] = peer_idx;
          send_op["outputChunk"] = peer_idx;
          send_op["peer"] = peer_rank;
          send_op["channel"] = 0;
          send_op["threadblock_count"] = tb_count;
          send_op["size"] = send_size;
          send_op["step"] = peer_idx;
          
          operations.push_back(send_op);
        }
        
        if (recv_size > 0) {  // Generate receive operation
          int tb_count = calculateThreadBlocks(recv_size);
          
          json recv_op;
          recv_op["type"] = "get";
          recv_op["inputChunk"] = peer_idx;
          recv_op["outputChunk"] = peer_idx;
          recv_op["peer"] = peer_rank;
          recv_op["channel"] = 0;
          recv_op["threadblock_count"] = tb_count;
          recv_op["size"] = recv_size;
          recv_op["step"] = peer_idx + params.peerRanks.size();
          
          operations.push_back(recv_op);
        }
      }
      
      gpu_json["operations"] = json::array({operations});
      gpus_json.push_back(gpu_json);
    }
  }
  
  concrete_json["gpus"] = gpus_json;
  
  return concrete_json.dump(2);
}

std::string DynamicExecutionPlan::createConcretePlan(const DynamicRuntimeParams& params, const std::string& outputPath) {
  std::string concrete_json = instantiate(params);
  
  std::ofstream file(outputPath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot create concrete execution plan file: " + outputPath);
  }
  
  file << concrete_json;
  file.close();
  
  return outputPath;
}

DynamicRuntimeParams DynamicAllToAllv::createRuntimeParams(
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes) {
  
  DynamicRuntimeParams params;
  
  // Calculate peer ranks (assume sequential for now)
  int num_ranks = std::max(sendSizes.size(), recvSizes.size());
  for (int i = 0; i < num_ranks; ++i) {
    params.peerRanks.push_back(i);
  }
  
  params.sendSizes = sendSizes;
  params.recvSizes = recvSizes;
  params.totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0UL);
  params.totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0UL);
  
  // Calculate offsets
  size_t send_offset = 0;
  size_t recv_offset = 0;
  for (size_t i = 0; i < sendSizes.size(); ++i) {
    params.sendOffsets.push_back(send_offset);
    send_offset += sendSizes[i];
  }
  for (size_t i = 0; i < recvSizes.size(); ++i) {
    params.recvOffsets.push_back(recv_offset);
    recv_offset += recvSizes[i];
  }
  
  params.maxThreadBlocks = 32;  // Default
  params.blockSize = 32768;     // Default
  
  return params;
}

bool DynamicAllToAllv::execute(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<DynamicExecutionPlan> dynamicPlan,
    void* /* sendBuffer */, void* /* recvBuffer */,
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes,
    int /* tag */) {
  
  if (!comm || !dynamicPlan) {
    return false;
  }
  
  try {
    // Create runtime parameters
    auto runtimeParams = createRuntimeParams(sendSizes, recvSizes);
    
    // Use the bootstrap to get the rank instead of comm->rank()
    int rank = comm->bootstrap()->getRank();
    
    // Generate concrete execution plan
    std::string concrete_plan_path = "/tmp/dynamic_alltoallv_" + 
                                   std::to_string(rank) + "_" + 
                                   std::to_string(std::time(nullptr)) + ".json";
    
    dynamicPlan->createConcretePlan(runtimeParams, concrete_plan_path);
    
    // TODO: Execute the concrete plan using MSCCLPP's execution engine
    // This would involve:
    // 1. Loading the concrete plan with ExecutionPlan
    // 2. Setting up the executor with the concrete plan
    // 3. Executing the all-to-allv operation
    
    // For now, just return success to indicate the dynamic plan was created
    return true;
    
  } catch (const std::exception& e) {
    return false;
  }
}

}  // namespace mscclpp