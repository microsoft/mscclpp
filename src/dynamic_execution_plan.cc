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
#include <chrono>

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

// Helper method for processing template variables - defined before use
void DynamicExecutionPlan::processJsonTemplateVariables(JsonType& json_obj, const VariableContext& var_context) {
  // Helper lambda to process a raw nlohmann::json recursively
  std::function<void(nlohmann::json&)> processRawJson = [&](nlohmann::json& j) {
    if (j.is_string()) {
      // If it's a string, substitute template variables
      std::string str_value = j.get<std::string>();
      std::string substituted = var_context.substituteVariables(str_value);
      
      // Try to convert to number if it looks like a number
      if (std::regex_match(substituted, std::regex(R"(^-?\d+$)"))) {
        j = std::stoll(substituted);
      } else if (std::regex_match(substituted, std::regex(R"(^-?\d*\.\d+$)"))) {
        j = std::stod(substituted);
      } else {
        j = substituted;
      }
    } else if (j.is_object()) {
      // Recursively process object members
      for (auto it = j.begin(); it != j.end(); ++it) {
        processRawJson(it.value());
      }
    } else if (j.is_array()) {
      // Recursively process array elements
      for (auto it = j.begin(); it != j.end(); ++it) {
        processRawJson(*it);
      }
    }
    // For other types (numbers, booleans, null), do nothing
  };
  
  // Process the JsonType object using the lambda
  nlohmann::json& raw_json = static_cast<nlohmann::json&>(json_obj);
  processRawJson(raw_json);
}

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
  
  std::cout << "Rank " << rank_ << ": Starting dynamic all-to-allv execution..." << std::endl;
  
  try {
    // Step 1: Create a concrete ExecutionPlan from the dynamic template
    auto executionPlan = plan_.createExecutionPlan(params);
    if (!executionPlan) {
      throw std::runtime_error("Failed to create concrete execution plan");
    }
    
    std::cout << "Rank " << rank_ << ": Created concrete ExecutionPlan: " << executionPlan->name() 
              << " (collective: " << executionPlan->collective() << ")" << std::endl;
    
    // Step 2: Execute using MSCCLPP executor
    size_t totalSendSize = params.totalSendSize;
    size_t totalRecvSize = params.totalRecvSize;
    
    std::cout << "Rank " << rank_ << ": Executing with total send size: " << totalSendSize 
              << ", total recv size: " << totalRecvSize << std::endl;
    
    // Step 3: Execute the concrete plan using the MSCCLPP executor
    // For alltoallv operations, we typically use FLOAT16 or UINT32 data type
    // Using UINT32 as it's suitable for variable-size data transfers
    executor->execute(
      rank_,                      // rank
      send_buff,                  // send buffer 
      recv_buff,                  // receive buffer
      totalSendSize,              // send buffer size
      totalRecvSize,              // receive buffer size
      mscclpp::DataType::UINT32,  // data type (can be adjusted based on actual data)
      *executionPlan,             // execution plan
      stream,                     // CUDA stream
      mscclpp::PacketType::LL16   // packet type (LL16 is commonly used)
    );
    
    std::cout << "Rank " << rank_ << ": Successfully executed dynamic all-to-allv using concrete ExecutionPlan" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Rank " << rank_ << ": Error during dynamic all-to-allv execution: " << e.what() << std::endl;
    
    // Fallback: Generate and print the concrete plan for debugging
    try {
      std::string concretePlan = plan_.instantiate(params);
      std::cout << "Rank " << rank_ << ": Generated concrete plan for debugging:\n" 
                << concretePlan.substr(0, 1000) << "..." << std::endl;  // Print first 1000 chars
    } catch (const std::exception& debug_e) {
      std::cout << "Rank " << rank_ << ": Failed to generate concrete plan for debugging: " 
                << debug_e.what() << std::endl;
    }
    
    throw;  // Re-throw the original exception
  }
}

std::unique_ptr<DynamicAllToAllv> DynamicExecutionPlan::createAllToAllv() {
  return std::make_unique<DynamicAllToAllv>(*this);
}

std::shared_ptr<ExecutionPlan> DynamicExecutionPlan::createExecutionPlan(const DynamicRuntimeParams& params) {
  // Instantiate the dynamic plan with runtime parameters
  std::string concretePlan = instantiate(params);
  
  // Create a temporary file to store the concrete plan
  std::string tempFileName = "/tmp/dynamic_plan_" + std::to_string(rank_) + "_" + 
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json";
  
  // Write the concrete plan to the temporary file
  std::ofstream outFile(tempFileName);
  if (!outFile.is_open()) {
    throw std::runtime_error("Cannot create temporary file: " + tempFileName);
  }
  
  outFile << concretePlan;
  outFile.close();
  
  // Store for cleanup (only store the latest one)
  if (!temp_file_path_.empty()) {
    std::remove(temp_file_path_.c_str());  // Remove previous temp file
  }
  temp_file_path_ = tempFileName;
  
  std::cout << "Rank " << rank_ << ": Created concrete execution plan file: " << tempFileName << std::endl;
  
  // Create and return the ExecutionPlan object
  try {
    auto executionPlan = std::make_shared<ExecutionPlan>(tempFileName, rank_);
    std::cout << "Rank " << rank_ << ": Successfully created ExecutionPlan from dynamic template" << std::endl;
    return executionPlan;
  } catch (const std::exception& e) {
    // Clean up the temp file if ExecutionPlan creation fails
    std::remove(tempFileName.c_str());
    temp_file_path_.clear();
    throw std::runtime_error("Failed to create ExecutionPlan: " + std::string(e.what()));
  }
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
  
  // Debug: Print the JSON structure
  std::cout << "Rank " << rank_ << ": Loaded JSON keys: ";
  for (auto& [key, value] : j.items()) {
    std::cout << key << "(" << (value.is_string() ? "string" : 
                                value.is_object() ? "object" : 
                                value.is_array() ? "array" : 
                                value.is_number() ? "number" : "other") << ") ";
  }
  std::cout << std::endl;
  
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
  
  // Parse dynamic parameters with better error handling
  if (j.contains("dynamic_parameters")) {
    std::cout << "Rank " << rank_ << ": Processing dynamic_parameters..." << std::endl;
    auto& dynamic_params = j["dynamic_parameters"];
    
    if (dynamic_params.is_object()) {
      for (auto& [key, value] : dynamic_params.items()) {
        if (value.is_string()) {
          dynamicParams_[key] = value.get<std::string>();
          std::cout << "Rank " << rank_ << ": Added dynamic param: " << key << " = " << value.get<std::string>() << std::endl;
        } else {
          std::cout << "Rank " << rank_ << ": Skipping non-string dynamic param: " << key 
                    << " (type: " << (value.is_object() ? "object" : 
                                     value.is_array() ? "array" : 
                                     value.is_number() ? "number" : "other") << ")" << std::endl;
        }
      }
    } else {
      std::cout << "Rank " << rank_ << ": Warning: dynamic_parameters is not an object" << std::endl;
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
    if (op["count"].is_string()) {
      std::string count_str = op["count"].get<std::string>();
      op["count"] = var_context.substituteVariables(count_str);
    }
  }
  
  if (op.contains("o_buff")) {
    if (op["o_buff"].is_string()) {
      std::string o_buff_str = op["o_buff"].get<std::string>();
      op["o_buff"] = var_context.substituteVariables(o_buff_str);
    }
  }
  
  if (op.contains("i_buff")) {
    if (op["i_buff"].is_string()) {
      std::string i_buff_str = op["i_buff"].get<std::string>();
      op["i_buff"] = var_context.substituteVariables(i_buff_str);
    }
  }
  
  if (op.contains("srcOffset")) {
    if (op["srcOffset"].is_string()) {
      std::string srcOffset_str = op["srcOffset"].get<std::string>();
      op["srcOffset"] = var_context.substituteVariables(srcOffset_str);
    }
  }
  
  if (op.contains("dstOffset")) {
    if (op["dstOffset"].is_string()) {
      std::string dstOffset_str = op["dstOffset"].get<std::string>();
      op["dstOffset"] = var_context.substituteVariables(dstOffset_str);
    }
  }
}

std::string DynamicExecutionPlan::instantiate(const DynamicRuntimeParams& params) {
  if (!templateJson_) {
    throw std::runtime_error("No template loaded");
  }
  
  std::cout << "Rank " << rank_ << ": Starting template instantiation..." << std::endl;
  
  try {
    std::cout << "Rank " << rank_ << ": Working directly with templateJson_..." << std::endl;
    // Work directly with the templateJson_ instead of creating a copy
    // This avoids the problematic copy constructor that's causing the error
    
    // Debug: Print the structure of the loaded template
    std::cout << "Rank " << rank_ << ": Analyzing template JSON structure..." << std::endl;
    for (auto& [key, value] : templateJson_->items()) {
      std::cout << "  " << key << ": " << (value.is_string() ? "string" :
                                           value.is_object() ? "object" :
                                           value.is_array() ? "array" :
                                           value.is_number() ? "number" : "other") << std::endl;
    }
    std::cout << "Rank " << rank_ << ": JSON structure analysis completed" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Starting variable context setup..." << std::endl;
    // Set up variable context with available DynamicRuntimeParams fields
    VariableContext var_context;
    
    std::cout << "Rank " << rank_ << ": Adding basic runtime parameters..." << std::endl;
    var_context.variables["num_ranks"] = std::to_string(params.num_ranks);
    var_context.variables["rank"] = std::to_string(rank_);
    var_context.variables["total_send_size"] = std::to_string(params.totalSendSize);
    var_context.variables["total_recv_size"] = std::to_string(params.totalRecvSize);
    var_context.variables["max_thread_blocks"] = std::to_string(params.maxThreadBlocks);
    var_context.variables["block_size"] = std::to_string(params.blockSize);
    
    std::cout << "Rank " << rank_ << ": Calculating thread blocks..." << std::endl;
    var_context.variables["thread_blocks"] = std::to_string(calculateThreadBlocks(params.totalSendSize));
    std::cout << "Rank " << rank_ << ": Thread blocks calculated successfully" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Adding dynamic template variables..." << std::endl;
    // Add the specific template variables that appear in the JSON
    var_context.variables["DYNAMIC_INPUT_CHUNKS"] = std::to_string(params.num_ranks);
    var_context.variables["DYNAMIC_OUTPUT_CHUNKS"] = std::to_string(params.num_ranks);
    var_context.variables["DYNAMIC_SCRATCH_CHUNKS"] = "0";
    
    // Add buffer-related template variables
    var_context.variables["src_buffer_type"] = "i";  // input buffer type
    var_context.variables["dst_buffer_type"] = "o";  // output buffer type
    std::cout << "Rank " << rank_ << ": Dynamic template variables added" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Adding default template variables..." << std::endl;
    // Add commonly used template variables with default values
    var_context.variables["src_chunk_index"] = "0";
    var_context.variables["src_chunk_size"] = "1024";
    var_context.variables["dst_chunk_index"] = "0";
    var_context.variables["dst_chunk_size"] = "1024";
    var_context.variables["chunk_size"] = "1024";
    var_context.variables["step_id"] = "0";
    var_context.variables["chunk_id"] = "0";
    var_context.variables["peer_rank"] = "0";
    var_context.variables["tb_count"] = "1";
    std::cout << "Rank " << rank_ << ": Default template variables added" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Adding per-peer variables..." << std::endl;
    // Add individual peer data for template substitution
    for (int i = 0; i < params.num_ranks; ++i) {
      var_context.variables["peer_rank_" + std::to_string(i)] = std::to_string(i);
      var_context.variables["channel_id_" + std::to_string(i)] = std::to_string(i);
      var_context.variables["chunk_id_" + std::to_string(i)] = std::to_string(i);
      var_context.variables["tb_count_" + std::to_string(i)] = std::to_string(1); // Default to 1 thread block
      
      // Add per-peer buffer variables
      var_context.variables["src_chunk_index_" + std::to_string(i)] = std::to_string(i * 1024);
      var_context.variables["dst_chunk_index_" + std::to_string(i)] = std::to_string(i * 1024);
      var_context.variables["src_chunk_size_" + std::to_string(i)] = "1024";
      var_context.variables["dst_chunk_size_" + std::to_string(i)] = "1024";
    }
    std::cout << "Rank " << rank_ << ": Per-peer variables added for " << params.num_ranks << " ranks" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Adding additional template variables..." << std::endl;
    // Add commonly used template variables
    var_context.variables["operation_type"] = "put"; // or "get", depending on operation
    var_context.variables["channel_id"] = "0";
    var_context.variables["src_buffer_id"] = "0";
    var_context.variables["dst_buffer_id"] = "0";
    var_context.variables["src_offset"] = "0";
    var_context.variables["dst_offset"] = "0";
    var_context.variables["count"] = "1024";
    std::cout << "Rank " << rank_ << ": Additional template variables added" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Generating size arrays..." << std::endl;
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
    std::cout << "Rank " << rank_ << ": Size arrays generated" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Variable context set up successfully with " 
              << var_context.variables.size() << " variables" << std::endl;
    
    // Debug: Print some of the variables
    std::cout << "Rank " << rank_ << ": Key variables: ";
    for (const auto& [key, value] : var_context.variables) {
      if (key.find("DYNAMIC") != std::string::npos || key == "chunk_size" || key == "peer_rank" || 
          key == "src_buffer_type" || key == "dst_buffer_type") {
        std::cout << key << "=" << value << " ";
      }
    }
    std::cout << std::endl;
    
    std::cout << "Rank " << rank_ << ": Starting template variable processing..." << std::endl;
    // Process template variables directly on the original templateJson_
    processJsonTemplateVariables(*templateJson_, var_context);
    std::cout << "Rank " << rank_ << ": Template variable processing completed" << std::endl;
    
    std::cout << "Rank " << rank_ << ": Generating final JSON..." << std::endl;
    // Generate the final JSON directly from templateJson_
    std::string result = templateJson_->dump(2);
    
    std::cout << "Rank " << rank_ << ": Template instantiation complete, result size: " 
              << result.length() << " characters" << std::endl;
    return result;
    
  } catch (const std::exception& e) {
    std::cout << "Rank " << rank_ << ": Error during instantiation: " << e.what() << std::endl;
    
    // Try to print some debug information about the template
    try {
      std::cout << "Rank " << rank_ << ": Template JSON dump (first 500 chars): " 
                << templateJson_->dump().substr(0, 500) << "..." << std::endl;
    } catch (...) {
      std::cout << "Rank " << rank_ << ": Could not dump template JSON for debugging" << std::endl;
    }
    
    throw;
  }
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
    if (operation_template["operation_type"].is_string()) {
      std::string op_type = operation_template["operation_type"].get<std::string>();
      operation_template["operation_type"] = var_context.substituteVariables(op_type);
    }
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
  
  // Debug: Print template structure for troubleshooting
  std::cout << "Rank " << rank_ << ": Processing operation template with keys: ";
  for (auto& [key, value] : operation_template.items()) {
    std::cout << key << "(" << (value.is_string() ? "string" : 
                                value.is_object() ? "object" : 
                                value.is_array() ? "array" : 
                                value.is_number() ? "number" : "other") << ") ";
  }
  std::cout << std::endl;
}

} // namespace mscclpp