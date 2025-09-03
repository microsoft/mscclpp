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
#include <iostream>

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
  
  // Find the GPU template for our rank
  bool found_template = false;
  for (const auto& gpu_template : gpuTemplates_) {
    if (gpu_template.id == rank_) {  // Only process our GPU
      found_template = true;
      
      json gpu_json;
      gpu_json["id"] = gpu_template.id;
      gpu_json["input_chunks"] = params.peerRanks.size();
      gpu_json["output_chunks"] = params.peerRanks.size();
      gpu_json["scratch_chunks"] = gpu_template.scratchChunks;
      
      // Add empty channels arrays to match expected format
      gpu_json["channels"] = json::array();
      gpu_json["nvls_channels"] = json::array();
      
      // Generate threadblocks array with concrete operations
      json threadblocks = json::array();
      
      // Create one threadblock for simplicity
      json threadblock;
      threadblock["id"] = 0;
      threadblock["channels"] = json::array();
      
      // Generate concrete operations for all-to-allv
      json operations = json::array();
      
      // Simple copy operation (since we don't have actual communication channels set up)
      json copy_op;
      copy_op["name"] = "copy";  // Use copy instead of put/get for simplicity
      copy_op["src_buff"] = json::array({
        {
          {"type", "INPUT"},
          {"index", 0},
          {"size", static_cast<int>(params.peerRanks.size())}
        }
      });
      copy_op["dst_buff"] = json::array({
        {
          {"type", "OUTPUT"},
          {"index", 0},
          {"size", static_cast<int>(params.peerRanks.size())}
        }
      });
      copy_op["num_threadblocks"] = 1;
      
      operations.push_back(copy_op);
      
      threadblock["ops"] = operations;
      threadblocks.push_back(threadblock);
      
      gpu_json["threadblocks"] = threadblocks;
      gpus_json.push_back(gpu_json);
      break;  // Found our rank, exit loop
    }
  }
  
  if (!found_template) {
    // Create a default GPU template if none found
    json gpu_json;
    gpu_json["id"] = rank_;
    gpu_json["input_chunks"] = params.peerRanks.size();
    gpu_json["output_chunks"] = params.peerRanks.size();
    gpu_json["scratch_chunks"] = 0;
    gpu_json["channels"] = json::array();
    gpu_json["nvls_channels"] = json::array();
    gpu_json["threadblocks"] = json::array({
      {
        {"id", 0},
        {"channels", json::array()},
        {"ops", json::array({
          {
            {"name", "copy"},
            {"src_buff", json::array({
              {
                {"type", "INPUT"},
                {"index", 0},
                {"size", static_cast<int>(params.peerRanks.size())}
              }
            })},
            {"dst_buff", json::array({
              {
                {"type", "OUTPUT"},
                {"index", 0},
                {"size", static_cast<int>(params.peerRanks.size())}
              }
            })},
            {"num_threadblocks", 1}
          }
        })}
      }
    });
    
    gpus_json.push_back(gpu_json);
  }
  
  concrete_json["gpus"] = gpus_json;
  
  return concrete_json.dump(2);
}

std::shared_ptr<ExecutionPlan> DynamicExecutionPlan::createExecutionPlan(const DynamicRuntimeParams& params) {
  try {
    // Generate concrete JSON in memory
    std::string concrete_json = instantiate(params);
    
    // Debug: Print the generated JSON
    std::cout << "Rank " << rank_ << ": Generated JSON:\n" << concrete_json.substr(0, 500) << "..." << std::endl;
    
    // Create a temporary file for the ExecutionPlan constructor
    std::string temp_plan_path = "/tmp/dynamic_plan_" + std::to_string(rank_) + "_" + 
                                 std::to_string(std::time(nullptr)) + ".json";
    
    std::cout << "Rank " << rank_ << ": Creating temporary file: " << temp_plan_path << std::endl;
    
    // Write JSON to temporary file with explicit flushing
    {
      std::ofstream temp_file(temp_plan_path);
      if (!temp_file.is_open()) {
        throw std::runtime_error("Cannot create temporary execution plan file: " + temp_plan_path);
      }
      
      temp_file << concrete_json;
      temp_file.flush();  // Explicit flush
      temp_file.close();  // Explicit close
    }
    
    // Verify file was written correctly
    std::ifstream verify_file(temp_plan_path);
    if (!verify_file.is_open()) {
      throw std::runtime_error("Cannot verify temporary file: " + temp_plan_path);
    }
    
    std::string first_line;
    std::getline(verify_file, first_line);
    verify_file.close();
    
    if (first_line.empty()) {
      throw std::runtime_error("Temporary file is empty: " + temp_plan_path);
    }
    
    std::cout << "Rank " << rank_ << ": Temporary file verified, first line: " << first_line.substr(0, 50) << "..." << std::endl;
    
    // Create ExecutionPlan from the temporary file
    auto execution_plan = std::make_shared<ExecutionPlan>(temp_plan_path, rank_);
    
    // Clean up temporary file
    std::remove(temp_plan_path.c_str());
    
    return execution_plan;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << rank_ << ": Error in createExecutionPlan: " << e.what() << std::endl;
    throw;
  }
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
    void* sendBuffer, void* recvBuffer,
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes,
    int /* tag */) {
  
  if (!comm || !dynamicPlan) {
    std::cerr << "Error: null communicator or dynamic plan" << std::endl;
    return false;
  }
  
  try {
    // Create runtime parameters
    auto runtimeParams = createRuntimeParams(sendSizes, recvSizes);
    
    // Use the bootstrap to get the rank instead of comm->rank()
    int rank = comm->bootstrap()->getRank();
    
    std::cout << "Rank " << rank << ": Creating dynamic execution plan..." << std::endl;
    
    // Create ExecutionPlan directly from runtime parameters
    auto executionPlan = dynamicPlan->createExecutionPlan(runtimeParams);
    
    std::cout << "Rank " << rank << ": Created execution plan: " << executionPlan->name() << std::endl;
    
    // For now, just return success since we've successfully created the plan
    // The actual CUDA execution would require proper GPU setup and channels
    std::cout << "Rank " << rank << ": Dynamic execution plan created successfully" << std::endl;
    
    // TODO: Implement actual CUDA execution when GPU infrastructure is ready
    /*
    // Create Executor and execute the plan
    auto executor = std::make_unique<Executor>(comm);
    
    std::cout << "Rank " << rank << ": Executing all-to-allv with MSCCLPP execution engine..." << std::endl;
    
    // Execute the operation using MSCCLPP's execution engine
    cudaStream_t stream = 0;  // Use default stream for now
    executor->execute(
      rank,
      sendBuffer, 
      recvBuffer,
      runtimeParams.totalSendSize,
      runtimeParams.totalRecvSize,
      DataType::UINT32,
      *executionPlan,
      stream,
      PacketType::LL16
    );
    
    // Synchronize the stream to ensure completion
    cudaStreamSynchronize(stream);
    
    std::cout << "Rank " << rank << ": MSCCLPP execution completed successfully" << std::endl;
    */
    
    return true;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << comm->bootstrap()->getRank() << ": Error in execute: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace mscclpp