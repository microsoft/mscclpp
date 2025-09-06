// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <regex>
#include <sstream>

namespace mscclpp {

// Define JsonType as a wrapper for nlohmann::json in the implementation
class DynamicExecutionPlan::JsonType : public nlohmann::json {
public:
  using nlohmann::json::json;  // Inherit constructors
  
  // Default constructor
  JsonType() : nlohmann::json() {}
  
  // Constructor from nlohmann::json
  JsonType(const nlohmann::json& j) : nlohmann::json(j) {}
  JsonType(nlohmann::json&& j) : nlohmann::json(std::move(j)) {}
  
  // Assignment operators from nlohmann::json
  JsonType& operator=(const nlohmann::json& j) {
    nlohmann::json::operator=(j);
    return *this;
  }
  
  JsonType& operator=(nlohmann::json&& j) {
    nlohmann::json::operator=(std::move(j));
    return *this;
  }
  
  // Implicit conversion to nlohmann::json
  operator nlohmann::json&() { return static_cast<nlohmann::json&>(*this); }
  operator const nlohmann::json&() const { return static_cast<const nlohmann::json&>(*this); }
};

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
      std::cout << "Warning: Variable ${" << var_name << "} not found in context" << std::endl;
    }
  }
  
  return result;
}

DynamicExecutionPlan::DynamicExecutionPlan(const std::string& planPath, int rank) 
  : rank_(rank), templateJson_(std::make_unique<JsonType>()) {
  loadFromJson(planPath);
}

DynamicExecutionPlan::~DynamicExecutionPlan() = default;

// DynamicAllToAllv implementation
DynamicAllToAllv::DynamicAllToAllv(DynamicExecutionPlan& plan) : plan_(plan), rank_(plan.getRank()) {}

DynamicRuntimeParams DynamicAllToAllv::createRuntimeParams(
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes) {
  
  DynamicRuntimeParams params;
  params.num_ranks = static_cast<int>(sendSizes.size());
  params.send_sizes = sendSizes;
  params.recv_sizes = recvSizes;
  
  // Calculate offsets
  params.send_offsets.resize(sendSizes.size());
  params.recv_offsets.resize(recvSizes.size());
  
  size_t sendOffset = 0;
  size_t recvOffset = 0;
  
  for (size_t i = 0; i < sendSizes.size(); ++i) {
    params.send_offsets[i] = sendOffset;
    sendOffset += sendSizes[i];
  }
  
  for (size_t i = 0; i < recvSizes.size(); ++i) {
    params.recv_offsets[i] = recvOffset;
    recvOffset += recvSizes[i];
  }
  
  params.totalSendSize = sendOffset;
  params.totalRecvSize = recvOffset;
  
  // Set default values for other parameters
  params.maxThreadBlocks = 32;
  params.blockSize = 32768;
  
  // Set peer ranks (for compatibility)
  params.peerRanks.resize(params.num_ranks);
  for (int i = 0; i < params.num_ranks; ++i) {
    params.peerRanks[i] = i;
  }
  
  return params;
}

bool DynamicAllToAllv::execute(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<DynamicExecutionPlan> dynamicPlan,
    void* sendBuffer, void* recvBuffer,
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes,
    int tag) {
  
  try {
    // Create runtime parameters
    auto params = createRuntimeParams(sendSizes, recvSizes);
    
    // Create executor
    auto executor = std::make_shared<Executor>(comm);
    
    // Create DynamicAllToAllv instance
    auto allToAllv = dynamicPlan->createAllToAllv();
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Execute the operation
    allToAllv->execute(
      sendBuffer, sendSizes, params.send_offsets,
      recvBuffer, recvSizes, params.recv_offsets,
      comm, executor, stream);
    
    // Synchronize and cleanup
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return true;
    
  } catch (const std::exception& e) {
    std::cout << "DynamicAllToAllv::execute failed: " << e.what() << std::endl;
    return false;
  }
}

void DynamicAllToAllv::execute(
    void* send_buff,
    const std::vector<size_t>& send_sizes,
    const std::vector<size_t>& send_offsets,
    void* recv_buff,
    const std::vector<size_t>& recv_sizes,
    const std::vector<size_t>& recv_offsets,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<Executor> executor,
    cudaStream_t stream) {
  
  // Create runtime parameters
  DynamicRuntimeParams params;
  params.num_ranks = static_cast<int>(send_sizes.size());
  params.send_sizes = send_sizes;
  params.recv_sizes = recv_sizes;
  params.send_offsets = send_offsets;
  params.recv_offsets = recv_offsets;
  params.totalSendSize = std::accumulate(send_sizes.begin(), send_sizes.end(), size_t(0));
  params.totalRecvSize = std::accumulate(recv_sizes.begin(), recv_sizes.end(), size_t(0));
  params.maxThreadBlocks = 32;
  params.blockSize = 32768;
  
  // Set peer ranks
  params.peerRanks.resize(params.num_ranks);
  for (int i = 0; i < params.num_ranks; ++i) {
    params.peerRanks[i] = i;
  }
  
  // Instantiate the dynamic plan with runtime parameters
  std::string concretePlan = plan_.instantiate(params);
  
  std::cout << "Rank " << rank_ << ": Generated concrete execution plan for dynamic all-to-allv" << std::endl;
  
  // For now, this is a placeholder implementation
  // In a full implementation, you would:
  // 1. Parse the concrete plan JSON
  // 2. Create the appropriate GPU operations
  // 3. Execute them using the provided executor and stream
  
  // TODO: Implement actual execution logic based on the concrete plan
  std::cout << "Rank " << rank_ << ": Dynamic all-to-allv execution completed (placeholder)" << std::endl;
}

std::unique_ptr<DynamicAllToAllv> DynamicExecutionPlan::createAllToAllv() {
  return std::make_unique<DynamicAllToAllv>(*this);
}

std::shared_ptr<ExecutionPlan> DynamicExecutionPlan::createExecutionPlan(const DynamicRuntimeParams& params) {
  // This would create a concrete ExecutionPlan from the instantiated template
  // For now, return nullptr as this requires more complex implementation
  std::string concretePlan = instantiate(params);
  // TODO: Parse concretePlan and create ExecutionPlan object
  return nullptr;
}

std::string DynamicExecutionPlan::createConcretePlan(const DynamicRuntimeParams& params, const std::string& outputPath) {
  std::string concretePlan = instantiate(params);
  
  // Write to file
  std::ofstream outFile(outputPath);
  if (!outFile.is_open()) {
    throw std::runtime_error("Cannot create output file: " + outputPath);
  }
  
  outFile << concretePlan;
  outFile.close();
  
  // Store for cleanup
  temp_file_path_ = outputPath;
  
  return outputPath;
}

void DynamicExecutionPlan::cleanup() {
  if (!temp_file_path_.empty()) {
    std::remove(temp_file_path_.c_str());
    temp_file_path_.clear();
  }
}

void DynamicExecutionPlan::loadFromJson(const std::string& planPath) {
  std::ifstream file(planPath);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open plan file: " + planPath);
  }
  
  nlohmann::json j;
  file >> j;
  
  // Store basic properties
  name_ = j.value("name", "");
  collective_ = j.value("collective", "");
  protocol_ = j.value("protocol", "");
  isDynamic_ = j.value("is_dynamic", true);
  minMessageSize_ = j.value("min_message_size", 0);
  maxMessageSize_ = j.value("max_message_size", SIZE_MAX);
  numThreadsPerBlock_ = j.value("num_threads_per_block", 256);
  
  std::cout << "Rank " << rank_ << ": Loaded DSL template: " << name_ 
            << ", collective: " << collective_ << ", protocol: " << protocol_ << std::endl;
  
  // Parse dynamic parameters
  if (j.contains("dynamic_parameters")) {
    for (auto& [key, value] : j["dynamic_parameters"].items()) {
      dynamicParams_[key] = value.get<std::string>();
    }
  }
  
  // Store the original template JSON for later instantiation
  *templateJson_ = j;
  
  std::cout << "Rank " << rank_ << ": Stored DSL template for runtime instantiation" << std::endl;
}

int DynamicExecutionPlan::calculateThreadBlocks(size_t messageSize) const {
  auto it = dynamicParams_.find("max_thread_blocks");
  int maxThreadBlocks = it != dynamicParams_.end() ? std::stoi(it->second) : 32;
  
  it = dynamicParams_.find("block_size");
  size_t blockSize = it != dynamicParams_.end() ? std::stoull(it->second) : 32768;
  
  return std::min(maxThreadBlocks, static_cast<int>((messageSize + blockSize - 1) / blockSize));
}

void DynamicExecutionPlan::updateOperationWithRuntimeParams(JsonType& op, 
                                                           const DynamicRuntimeParams& params, 
                                                           const VariableContext& var_context) {
  // Template substitution for operation parameters
  if (op.contains("count")) {
    std::string count_str = op["count"].get<std::string>();
    op["count"] = var_context.substituteVariables(count_str);
  }
  
  if (op.contains("o_buff")) {
    std::string o_buff_str = op["o_buff"].get<std::string>();
    op["o_buff"] = var_context.substituteVariables(o_buff_str);
  }
  
  if (op.contains("i_buff")) {
    std::string i_buff_str = op["i_buff"].get<std::string>();
    op["i_buff"] = var_context.substituteVariables(i_buff_str);
  }
  
  if (op.contains("srcOffset")) {
    std::string srcOffset_str = op["srcOffset"].get<std::string>();
    op["srcOffset"] = var_context.substituteVariables(srcOffset_str);
  }
  
  if (op.contains("dstOffset")) {
    std::string dstOffset_str = op["dstOffset"].get<std::string>();
    op["dstOffset"] = var_context.substituteVariables(dstOffset_str);
  }
}

std::string DynamicExecutionPlan::instantiate(const DynamicRuntimeParams& params) {
  if (!templateJson_) {
    throw std::runtime_error("No template loaded");
  }
  
  // Create a working copy of the template
  nlohmann::json concrete_json = *templateJson_;
  
  // Set up variable context with available DynamicRuntimeParams fields
  VariableContext var_context;
  var_context.variables["num_ranks"] = std::to_string(params.num_ranks);
  var_context.variables["rank"] = std::to_string(rank_);
  var_context.variables["total_send_size"] = std::to_string(params.totalSendSize);
  var_context.variables["total_recv_size"] = std::to_string(params.totalRecvSize);
  var_context.variables["max_thread_blocks"] = std::to_string(params.maxThreadBlocks);
  var_context.variables["block_size"] = std::to_string(params.blockSize);
  var_context.variables["thread_blocks"] = std::to_string(calculateThreadBlocks(params.totalSendSize));
  
  // Add send/recv sizes as comma-separated strings for template use
  std::stringstream send_sizes_str, recv_sizes_str, send_offsets_str, recv_offsets_str;
  for (size_t i = 0; i < params.send_sizes.size(); ++i) {
    if (i > 0) {
      send_sizes_str << ",";
      recv_sizes_str << ",";
      send_offsets_str << ",";
      recv_offsets_str << ",";
    }
    send_sizes_str << params.send_sizes[i];
    recv_sizes_str << params.recv_sizes[i];
    send_offsets_str << params.send_offsets[i];
    recv_offsets_str << params.recv_offsets[i];
  }
  var_context.variables["send_sizes"] = send_sizes_str.str();
  var_context.variables["recv_sizes"] = recv_sizes_str.str();
  var_context.variables["send_offsets"] = send_offsets_str.str();
  var_context.variables["recv_offsets"] = recv_offsets_str.str();
  
  std::cout << "Rank " << rank_ << ": Instantiating template with total_send_size=" << params.totalSendSize
            << ", total_recv_size=" << params.totalRecvSize << ", num_ranks=" << params.num_ranks << std::endl;
  
  // Update GPU-specific sections
  if (concrete_json.contains("gpus") && rank_ < static_cast<int>(concrete_json["gpus"].size())) {
    auto& gpu_json = concrete_json["gpus"][rank_];
    
    // Process threadblocks and operations
    if (gpu_json.contains("threadblocks")) {
      for (auto& threadblock : gpu_json["threadblocks"]) {
        if (threadblock.contains("ops")) {
          for (auto& op : threadblock["ops"]) {
            // Update operations marked as templates
            if (op.contains("template") && op["template"].get<bool>()) {
              // Convert nlohmann::json to JsonType for the method call
              JsonType op_wrapper(op);
              updateOperationWithRuntimeParams(op_wrapper, params, var_context);
              op = static_cast<nlohmann::json>(op_wrapper);  // Copy back
            }
          }
        }
      }
    }
    
    // Process operation templates
    JsonType gpu_wrapper(gpu_json);
    processOperationTemplates(gpu_wrapper, params, var_context);
    gpu_json = static_cast<nlohmann::json>(gpu_wrapper);  // Copy back
    
    std::cout << "Rank " << rank_ << ": Updated DSL JSON with runtime parameters" << std::endl;
  }
  
  // For simplicity in this example, create a local copy-only version
  std::string result = concrete_json.dump(2);
  
  std::cout << "Rank " << rank_ << ": Template instantiation complete" << std::endl;
  return result;
}

void DynamicExecutionPlan::processOperationTemplates(JsonType& gpu_json, 
                                                    const DynamicRuntimeParams& params, 
                                                    const VariableContext& var_context) {
  if (!gpu_json.contains("operations")) {
    return;
  }
  
  auto& operations = gpu_json["operations"];
  for (auto& operation : operations) {
    if (operation.contains("operation_template")) {
      auto& operation_template = operation["operation_template"];
      JsonType template_wrapper(operation_template);
      substituteOperationTemplateVariables(template_wrapper, params, var_context);
      operation_template = static_cast<nlohmann::json>(template_wrapper);  // Copy back
    }
  }
}

void DynamicExecutionPlan::substituteOperationTemplateVariables(JsonType& operation_template,
                                                               const DynamicRuntimeParams& params,
                                                               const VariableContext& var_context) {
  // Enhanced template variable substitution for operation templates
  
  // Handle operation_type substitution
  if (operation_template.contains("operation_type")) {
    std::string op_type = operation_template["operation_type"].get<std::string>();
    operation_template["operation_type"] = var_context.substituteVariables(op_type);
  }
  
  // Handle channel_id substitution  
  if (operation_template.contains("channel_id")) {
    if (operation_template["channel_id"].is_string()) {
      std::string channel_id = operation_template["channel_id"].get<std::string>();
      operation_template["channel_id"] = var_context.substituteVariables(channel_id);
    }
  }
  
  // Handle peer_rank substitution
  if (operation_template.contains("peer_rank")) {
    if (operation_template["peer_rank"].is_string()) {
      std::string peer_rank = operation_template["peer_rank"].get<std::string>();
      operation_template["peer_rank"] = var_context.substituteVariables(peer_rank);
    }
  }
  
  // Handle chunk_id substitution
  if (operation_template.contains("chunk_id")) {
    if (operation_template["chunk_id"].is_string()) {
      std::string chunk_id = operation_template["chunk_id"].get<std::string>();
      operation_template["chunk_id"] = var_context.substituteVariables(chunk_id);
    }
  }
  
  // Handle tb_count substitution
  if (operation_template.contains("tb_count")) {
    if (operation_template["tb_count"].is_string()) {
      std::string tb_count = operation_template["tb_count"].get<std::string>();
      operation_template["tb_count"] = var_context.substituteVariables(tb_count);
    }
  }
  
  // Handle src_buffer_id substitution
  if (operation_template.contains("src_buffer_id")) {
    if (operation_template["src_buffer_id"].is_string()) {
      std::string src_buffer_id = operation_template["src_buffer_id"].get<std::string>();
      operation_template["src_buffer_id"] = var_context.substituteVariables(src_buffer_id);
    }
  }
  
  // Handle dst_buffer_id substitution
  if (operation_template.contains("dst_buffer_id")) {
    if (operation_template["dst_buffer_id"].is_string()) {
      std::string dst_buffer_id = operation_template["dst_buffer_id"].get<std::string>();
      operation_template["dst_buffer_id"] = var_context.substituteVariables(dst_buffer_id);
    }
  }
  
  // Handle src_offset substitution
  if (operation_template.contains("src_offset")) {
    if (operation_template["src_offset"].is_string()) {
      std::string src_offset = operation_template["src_offset"].get<std::string>();
      operation_template["src_offset"] = var_context.substituteVariables(src_offset);
    }
  }
  
  // Handle dst_offset substitution
  if (operation_template.contains("dst_offset")) {
    if (operation_template["dst_offset"].is_string()) {
      std::string dst_offset = operation_template["dst_offset"].get<std::string>();
      operation_template["dst_offset"] = var_context.substituteVariables(dst_offset);
    }
  }
  
  // Handle count substitution
  if (operation_template.contains("count")) {
    if (operation_template["count"].is_string()) {
      std::string count = operation_template["count"].get<std::string>();
      operation_template["count"] = var_context.substituteVariables(count);
    }
  }
  
  // Handle any nested operation_template objects recursively
  for (auto& [key, value] : operation_template.items()) {
    if (value.is_object() && key == "operation_template") {
      JsonType nested_template(value);
      substituteOperationTemplateVariables(nested_template, params, var_context);
      value = static_cast<nlohmann::json>(nested_template);
    }
  }
}

} // namespace mscclpp