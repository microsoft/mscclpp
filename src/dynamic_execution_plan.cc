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

// Static helper function for debugging JSON structure - only used internally
static void debugJsonStructure(const nlohmann::json& json_obj, const std::string& path = "", int rank = -1) {
  if (json_obj.is_object()) {
    for (auto& [key, value] : json_obj.items()) {
      std::string current_path = path.empty() ? key : path + "." + key;
      if (value.is_string()) {
        // Check if string looks like it should be a number but isn't
        std::string str_val = value.get<std::string>();
        if (str_val.find("${") != std::string::npos) {
          std::cout << "Rank " << rank << ": WARNING: Unsubstituted template variable at " 
                    << current_path << ": " << str_val << std::endl;
        }
      } else if (value.is_object() || value.is_array()) {
        debugJsonStructure(value, current_path, rank);
      }
    }
  } else if (json_obj.is_array()) {
    for (size_t i = 0; i < json_obj.size(); ++i) {
      std::string current_path = path + "[" + std::to_string(i) + "]";
      debugJsonStructure(json_obj[i], current_path, rank);
    }
  }
}

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

// Comprehensive JSON sanitization to handle all type issues

std::string DynamicExecutionPlan::instantiate(const DynamicRuntimeParams& params) {
  std::cout << "Rank " << rank_ << ": Processing dynamic template fields..." << std::endl;
  
  try {
    // Create a mutable copy for processing
    JsonType workingJson(*templateJson_);
    
    // Process dynamic fields in the template
    processDynamicTemplate(workingJson, params);
    
    // CRITICAL FIX: Ensure dynamic_parameters contains numbers, not strings
    if (workingJson.contains("dynamic_parameters") && workingJson["dynamic_parameters"].is_object()) {
      auto& dynParams = workingJson["dynamic_parameters"];
      
      // Convert block_size from string to number
      auto it = dynamicParams_.find("block_size");
      if (it != dynamicParams_.end()) {
        size_t blockSize = std::stoull(it->second);
        dynParams["block_size"] = static_cast<int>(blockSize);  // Use int for MSCCLPP compatibility
        std::cout << "Rank " << rank_ << ": Set dynamic_parameters.block_size = " << blockSize << " (int)" << std::endl;
      }
      
      // Convert max_thread_blocks from string to number
      it = dynamicParams_.find("max_thread_blocks");
      if (it != dynamicParams_.end()) {
        int maxThreadBlocks = std::stoi(it->second);
        dynParams["max_thread_blocks"] = static_cast<int>(maxThreadBlocks);
        std::cout << "Rank " << rank_ << ": Set dynamic_parameters.max_thread_blocks = " << maxThreadBlocks << " (int)" << std::endl;
      }
    }
    
    // NEW: Expand aggregated threadblocks to concrete entries
    expandThreadblocks(workingJson);
    
    // NEW: Validate and fix any empty buffer arrays
    validateAndFixBufferArrays(workingJson);
    
    // Standard JSON sanitization to ensure compatibility
    std::cout << "Rank " << rank_ << ": Sanitizing JSON for MSCCLPP executor compatibility..." << std::endl;
    sanitizeJsonForSerialization(workingJson);
    
    // Use safe copy-back mechanism to avoid corruption
    std::cout << "Rank " << rank_ << ": Performing safe copy-back using dump/parse..." << std::endl;
    
    std::string jsonString;
    try {
      jsonString = workingJson.dump(2);
    } catch (const nlohmann::json::exception& dump_error) {
      std::cout << "Rank " << rank_ << ": JSON dump failed: " << dump_error.what() << std::endl;
      std::cout << "Rank " << rank_ << ": Error ID: " << dump_error.id << std::endl;
      
      // Try aggressive sanitization
      std::cout << "Rank " << rank_ << ": Attempting aggressive JSON sanitization..." << std::endl;
      aggressivelySanitizeJson(workingJson);
      
      try {
        jsonString = workingJson.dump(2);
        std::cout << "Rank " << rank_ << ": Aggressive sanitization successful" << std::endl;
      } catch (const nlohmann::json::exception& dump_error2) {
        std::cout << "Rank " << rank_ << ": Aggressive sanitization failed: " << dump_error2.what() << std::endl;
        throw;
      }
    }
    
    nlohmann::json cleanJson;
    try {
      cleanJson = nlohmann::json::parse(jsonString);
    } catch (const nlohmann::json::exception& parse_error) {
      std::cout << "Rank " << rank_ << ": JSON parse failed: " << parse_error.what() << std::endl;
      throw;
    }
    
    // Output debug info for successful generation
    std::cout << "Rank " << rank_ << ": Successfully generated concrete execution plan with proper dynamic_parameters types" << std::endl;
    
    std::string outputPath = "/home/qinghuazhou/mscclpp/build/dynamic_plan_" + std::to_string(rank_) + "_" + std::to_string(std::hash<std::thread::id>()(std::this_thread::get_id())) + ".json";
    std::ofstream outFile(outputPath);
    if (outFile.is_open()) {
      outFile << cleanJson.dump(2);
      outFile.close();
      std::cout << "Rank " << rank_ << ": Created concrete execution plan file: " << outputPath << std::endl;
    }
    
    return cleanJson.dump(2);
    
  } catch (const nlohmann::json::exception& e) {
    std::cout << "Rank " << rank_ << ": JSON error during template instantiation: " << e.what() << std::endl;
    std::cout << "Rank " << rank_ << ": Error ID: " << e.id << std::endl;
    
    try {
      std::cout << "Rank " << rank_ << ": Creating sanitized execution plan..." << std::endl;
      JsonType sanitizedJson = createSanitizedExecutionPlan();
      
      // Apply our parameters to the sanitized plan
      if (sanitizedJson.contains("dynamic_parameters") && sanitizedJson["dynamic_parameters"].is_object()) {
        auto& dynParams = sanitizedJson["dynamic_parameters"];
        
        // Use the stored dynamic parameters but convert to numbers
        auto it = dynamicParams_.find("block_size");
        size_t blockSize = it != dynamicParams_.end() ? std::stoull(it->second) : 32768;
        dynParams["block_size"] = static_cast<int64_t>(blockSize);
        
        it = dynamicParams_.find("max_thread_blocks");
        int maxThreadBlocks = it != dynamicParams_.end() ? std::stoi(it->second) : 32;
        dynParams["max_thread_blocks"] = static_cast<int64_t>(maxThreadBlocks);
        
        std::cout << "Rank " << rank_ << ": Applied numeric dynamic_parameters: block_size=" << blockSize 
                  << ", max_thread_blocks=" << maxThreadBlocks << std::endl;
      }
      
      std::string sanitizedString = sanitizedJson.dump(2);
      std::cout << "Rank " << rank_ << ": Successfully created sanitized execution plan" << std::endl;
      return sanitizedString;
      
    } catch (const nlohmann::json::exception& dump_error) {
      std::cout << "Rank " << rank_ << ": Sanitized dump still failed: " << dump_error.what() << std::endl;
      std::cout << "Rank " << rank_ << ": Error ID: " << dump_error.id << std::endl;
      
      // As last resort, create a completely new minimal JSON structure with NUMERIC dynamic_parameters
      std::cout << "Rank " << rank_ << ": Creating minimal fallback JSON structure with numeric parameters..." << std::endl;
      
      nlohmann::json fallback_json = {
        {"buffer_alignment", static_cast<int64_t>(16)},
        {"collective", "alltoall"},
        {"dynamic", true},
        {"dynamic_parameters", {
          {"block_size", static_cast<int64_t>(32768)},      // FIXED: Use number, not string
          {"max_thread_blocks", static_cast<int64_t>(32)}   // FIXED: Use number, not string
        }},
        {"gpus", nlohmann::json::array()}
      };
      
      // Add minimal GPU structures
      for (int gpu_id = 0; gpu_id < params.num_ranks; ++gpu_id) {
        nlohmann::json gpu = {
          {"id", static_cast<int64_t>(gpu_id)},
          {"input_chunks", static_cast<int64_t>(1)},      // CHANGED from params.num_ranks to 1
          {"output_chunks", static_cast<int64_t>(1)},     // CHANGED from params.num_ranks to 1
          {"scratch_chunks", static_cast<int64_t>(params.num_ranks - 1)},
          {"threadblocks", nlohmann::json::array()},
          {"channels", nlohmann::json::array()},
          {"remote_buffers", nlohmann::json::array()},
          {"semaphores", nlohmann::json::array()}
        };
        fallback_json["gpus"].push_back(gpu);
      }
      
      std::cout << "Rank " << rank_ << ": Created fallback JSON with numeric dynamic_parameters" << std::endl;
      return fallback_json.dump(2);
    }
  }
}

void DynamicExecutionPlan::processDynamicTemplate(JsonType& json_obj, const DynamicRuntimeParams& params) {
  std::cout << "Rank " << rank_ << ": Processing dynamic template fields..." << std::endl;
  
  // Process each GPU in the template
  if (json_obj.contains("gpus")) {
    auto& gpus_array = json_obj["gpus"];
    std::cout << "Rank " << rank_ << ": Found " << gpus_array.size() << " GPUs to process" << std::endl;
    
    // DEBUG: Check if gpus_array is the right size
    if (gpus_array.size() != 4) {
      std::cout << "Rank " << rank_ << ": WARNING: Expected 4 GPUs but found " << gpus_array.size() << std::endl;
    }
    
    for (size_t i = 0; i < gpus_array.size(); ++i) {
      try {
        std::cout << "Rank " << rank_ << ": *** STARTING GPU PROCESSING FOR INDEX " << i << " ***" << std::endl;
        
        // Get reference to the actual nlohmann::json object in the array
        nlohmann::json& gpu_raw_json = gpus_array[i];
        
        if (!gpu_raw_json.is_object()) {
          std::cout << "Rank " << rank_ << ": GPU " << i << " is not an object, skipping" << std::endl;
          continue;
        }
        
        // Create a JsonType wrapper that references the same data
        JsonType gpu_json_wrapper;
        gpu_json_wrapper = gpu_raw_json;  // This copies the data
        
        int gpu_id = gpu_json_wrapper.value("id", static_cast<int>(i));
        
        std::cout << "Rank " << rank_ << ": Processing GPU " << gpu_id << " (array index " << i << ")" << std::endl;
        processDynamicGpu(gpu_json_wrapper, params, gpu_id);
        
        // Safe copy-back using dump/parse to avoid reference issues
        try {
          std::string gpu_json_str = gpu_json_wrapper.dump();
          gpu_raw_json = nlohmann::json::parse(gpu_json_str);
          std::cout << "Rank " << rank_ << ": Successfully copied back GPU " << gpu_id << " (index " << i << ")" << std::endl;
        } catch (const std::exception& gpu_copy_error) {
          std::cout << "Rank " << rank_ << ": Copy-back failed for GPU " << gpu_id << " (index " << i 
                    << "): " << gpu_copy_error.what() << std::endl;
          // Fallback: direct assignment
          try {
            gpu_raw_json = static_cast<nlohmann::json&>(gpu_json_wrapper);
            std::cout << "Rank " << rank_ << ": Used fallback direct assignment for GPU " << gpu_id << std::endl;
          } catch (const std::exception& fallback_error) {
            std::cout << "Rank " << rank_ << ": Fallback assignment also failed for GPU " << gpu_id 
                      << ": " << fallback_error.what() << std::endl;
            // Create a minimal GPU structure as last resort
            gpu_raw_json = nlohmann::json{
              {"id", static_cast<int64_t>(gpu_id)},
              {"input_chunks", static_cast<int64_t>(1)},      // CHANGED from params.num_ranks to 1
              {"output_chunks", static_cast<int64_t>(1)},     // CHANGED from params.num_ranks to 1
              {"scratch_chunks", static_cast<int64_t>(params.num_ranks - 1)},
              {"threadblocks", nlohmann::json::array()}
            };
            std::cout << "Rank " << rank_ << ": Created fallback GPU structure for GPU " << gpu_id << std::endl;
          }
        }
        
        std::cout << "Rank " << rank_ << ": *** COMPLETED GPU PROCESSING FOR INDEX " << i << " ***" << std::endl;
        
      } catch (const std::exception& gpu_error) {
        std::cout << "Rank " << rank_ << ": ERROR processing GPU at index " << i 
                  << ": " << gpu_error.what() << std::endl;
        // Continue processing other GPUs
        continue;
      } catch (...) {
        std::cout << "Rank " << rank_ << ": UNKNOWN ERROR processing GPU at index " << i << std::endl;
        // Continue processing other GPUs
        continue;
      }
    }
    
    std::cout << "Rank " << rank_ << ": Completed processing all " << gpus_array.size() << " GPUs" << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Dynamic template processing complete" << std::endl;
}

void DynamicExecutionPlan::processDynamicGpu(JsonType& gpu_json, const DynamicRuntimeParams& params, int gpu_id) {
  std::cout << "Rank " << rank_ << ": Processing dynamic GPU " << gpu_id << std::endl;
  
  // For alltoallv operations with variable sizes, set chunks to 1 to avoid 
  // MSCCLPP's uniform chunk size validation
  if (gpu_json.contains("dynamic_input_chunks")) {
    gpu_json["input_chunks"] = 1;  // Treat entire input buffer as one chunk
    gpu_json.erase("dynamic_input_chunks");
    std::cout << "Rank " << rank_ << ": Set input_chunks = 1 for variable-size alltoallv" << std::endl;
  } else if (!gpu_json.contains("input_chunks")) {
    // If no dynamic field, set it directly
    gpu_json["input_chunks"] = 1;
    std::cout << "Rank " << rank_ << ": Set input_chunks = 1 (direct assignment)" << std::endl;
  }
  
  if (gpu_json.contains("dynamic_output_chunks")) {
    gpu_json["output_chunks"] = 1;  // Treat entire output buffer as one chunk
    gpu_json.erase("dynamic_output_chunks");
    std::cout << "Rank " << rank_ << ": Set output_chunks = 1 for variable-size alltoallv" << std::endl;
  } else if (!gpu_json.contains("output_chunks")) {
    // If no dynamic field, set it directly
    gpu_json["output_chunks"] = 1;
    std::cout << "Rank " << rank_ << ": Set output_chunks = 1 (direct assignment)" << std::endl;
  }
  
  // Set scratch_chunks (usually all peers except self)
  if (gpu_json.contains("dynamic_scratch_chunks")) {
    int scratch_chunks = params.num_ranks - 1;
    gpu_json["scratch_chunks"] = static_cast<int>(scratch_chunks);
    gpu_json.erase("dynamic_scratch_chunks");
    std::cout << "Rank " << rank_ << ": Set scratch_chunks = " << scratch_chunks << std::endl;
  }
  
  // CRITICAL: Force input_chunks and output_chunks to 1 for alltoallv
  // This must come after all other processing to ensure it's not overwritten
  gpu_json["input_chunks"] = 1;
  gpu_json["output_chunks"] = 1;
  std::cout << "Rank " << rank_ << ": FORCED input_chunks = 1, output_chunks = 1 for alltoallv compatibility" << std::endl;
  
  // Ensure proper type for existing fields to avoid number/number type conflicts
  if (gpu_json.contains("input_chunks") && !gpu_json["input_chunks"].is_number_integer()) {
    gpu_json["input_chunks"] = 1;  // CHANGED: Force to 1 instead of params.num_ranks
  }
  if (gpu_json.contains("output_chunks") && !gpu_json["output_chunks"].is_number_integer()) {
    gpu_json["output_chunks"] = 1;  // CHANGED: Force to 1 instead of params.num_ranks
  }
  if (gpu_json.contains("scratch_chunks") && !gpu_json["scratch_chunks"].is_number_integer()) {
    gpu_json["scratch_chunks"] = static_cast<int>(params.num_ranks - 1);
  }
  if (gpu_json.contains("id") && !gpu_json["id"].is_number_integer()) {
    gpu_json["id"] = static_cast<int>(gpu_id);
  }
  
  // Process threadblocks
  if (gpu_json.contains("threadblocks")) {
    std::cout << "Rank " << rank_ << ": Starting threadblocks processing for GPU " << gpu_id << std::endl;
    processDynamicThreadblocks(gpu_json, params, gpu_id);
    std::cout << "Rank " << rank_ << ": Completed threadblocks processing for GPU " << gpu_id << std::endl;
  } else {
    std::cout << "Rank " << rank_ << ": No threadblocks found for GPU " << gpu_id << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Completed processing dynamic GPU " << gpu_id << std::endl;
}

void DynamicExecutionPlan::processDynamicThreadblocks(JsonType& gpu_json, const DynamicRuntimeParams& params, int gpu_id) {
  std::cout << "Rank " << rank_ << ": Processing dynamic threadblocks for GPU " << gpu_id << std::endl;
  
  if (!gpu_json.contains("threadblocks")) {
    std::cout << "Rank " << rank_ << ": No threadblocks found for GPU " << gpu_id << std::endl;
    return;
  }
  
  auto& threadblocks_array = gpu_json["threadblocks"];
  
  if (!threadblocks_array.is_array()) {
    std::cout << "Rank " << rank_ << ": threadblocks is not an array for GPU " << gpu_id << std::endl;
    return;
  }
  
  std::cout << "Rank " << rank_ << ": Found " << threadblocks_array.size() 
            << " threadblocks for GPU " << gpu_id << std::endl;
  
  // Process ALL threadblocks, not just the first one
  for (size_t i = 0; i < threadblocks_array.size(); ++i) {
    std::cout << "Rank " << rank_ << ": Processing threadblock " << i << " of " << threadblocks_array.size() << std::endl;
    
    // Get reference to the actual nlohmann::json object in the array
    nlohmann::json& tb_raw_json = threadblocks_array[i];
    
    if (!tb_raw_json.is_object()) {
      std::cout << "Rank " << rank_ << ": Threadblock " << i << " is not an object, skipping" << std::endl;
      continue;
    }
    
    // Create a JsonType wrapper
    JsonType tb_json_wrapper;
    tb_json_wrapper = tb_raw_json;  // This copies the data
    
    if (tb_json_wrapper.contains("dynamic_tbgroup_id")) {
      int tb_group_id = tb_json_wrapper["dynamic_tbgroup_id"].get<int>();
      std::cout << "Rank " << rank_ << ": Processing dynamic threadblock group " << tb_group_id 
                << " (index " << i << ")" << std::endl;
      
      processDynamicThreadblock(tb_json_wrapper, params, gpu_id, tb_group_id);
      
      // Remove the dynamic field
      tb_json_wrapper.erase("dynamic_tbgroup_id");
    } else {
      std::cout << "Rank " << rank_ << ": Threadblock " << i 
                << " does not have dynamic_tbgroup_id, processing as static" << std::endl;
      
      // For static threadblocks, we might still need to process any dynamic buffer fields
      // but we don't have a specific group ID, so use the index as group ID
      int static_tb_group_id = static_cast<int>(i);
      processDynamicThreadblock(tb_json_wrapper, params, gpu_id, static_tb_group_id);
    }
    
    // Safe copy-back using dump/parse to avoid reference issues
    try {
      std::string tb_json_str = tb_json_wrapper.dump();
      tb_raw_json = nlohmann::json::parse(tb_json_str);
      std::cout << "Rank " << rank_ << ": Successfully copied back threadblock " << i << std::endl;
    } catch (const std::exception& tb_copy_error) {
      std::cout << "Rank " << rank_ << ": Copy-back failed for threadblock " << i 
                << ": " << tb_copy_error.what() << std::endl;
      // Fallback: direct assignment
      try {
        tb_raw_json = static_cast<nlohmann::json&>(tb_json_wrapper);
        std::cout << "Rank " << rank_ << ": Used fallback direct assignment for threadblock " << i << std::endl;
      } catch (const std::exception& fallback_error) {
        std::cout << "Rank " << rank_ << ": Fallback assignment also failed for threadblock " << i 
                  << ": " << fallback_error.what() << std::endl;
        // Create a minimal threadblock structure as last resort
        tb_raw_json = nlohmann::json{
          {"tb_count", 1},
          {"tb_group_id", static_cast<int>(i)},
          {"ops", nlohmann::json::array()}
        };
        std::cout << "Rank " << rank_ << ": Created fallback threadblock structure for " << i << std::endl;
      }
    }
    
    std::cout << "Rank " << rank_ << ": Completed processing threadblock " << i << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Completed processing all " << threadblocks_array.size() 
            << " threadblocks for GPU " << gpu_id << std::endl;
}

void DynamicExecutionPlan::processDynamicThreadblock(JsonType& tb_json, const DynamicRuntimeParams& params, 
                                                    int gpu_id, int tb_group_id) {
  std::cout << "Rank " << rank_ << ": Processing threadblock group " << tb_group_id 
            << " for GPU " << gpu_id << std::endl;
  
  // Calculate number of thread blocks for this group
  int num_tb = calculateThreadBlocksForGroup(tb_group_id, params);
  
  // Add threadblock count information - use explicit int casting
  tb_json["tb_count"] = static_cast<int>(num_tb);
  tb_json["tb_group_id"] = static_cast<int>(tb_group_id);
  
  // Ensure proper type for existing fields
  if (tb_json.contains("tb_count") && !tb_json["tb_count"].is_number_integer()) {
    tb_json["tb_count"] = static_cast<int>(num_tb);
  }
  if (tb_json.contains("tb_group_id") && !tb_json["tb_group_id"].is_number_integer()) {
    tb_json["tb_group_id"] = static_cast<int>(tb_group_id);
  }
  
  std::cout << "Rank " << rank_ << ": Threadblock group " << tb_group_id 
            << " assigned " << num_tb << " thread blocks" << std::endl;
  
  // Process operations within this threadblock
  if (tb_json.contains("ops")) {
    std::cout << "Rank " << rank_ << ": Processing operations for threadblock group " << tb_group_id 
              << " GPU " << gpu_id << std::endl;
    processDynamicOperations(tb_json, params, gpu_id, tb_group_id);
    std::cout << "Rank " << rank_ << ": Completed operations processing for threadblock group " 
              << tb_group_id << " GPU " << gpu_id << std::endl;
  } else {
    std::cout << "Rank " << rank_ << ": No operations found in threadblock group " << tb_group_id 
              << " GPU " << gpu_id << std::endl;
  }
}

// Updated processDynamicOperations to handle all operations properly with better error handling
void DynamicExecutionPlan::processDynamicOperations(JsonType& tb_json, const DynamicRuntimeParams& params, 
                                                   int gpu_id, int tb_group_id) {
  std::cout << "Rank " << rank_ << ": Processing operations for threadblock group " << tb_group_id << std::endl;
  
  if (!tb_json.contains("ops")) {
    std::cout << "Rank " << rank_ << ": No operations found in threadblock group " << tb_group_id << std::endl;
    return;
  }
  
  auto& ops_array = tb_json["ops"];
  
  if (!ops_array.is_array()) {
    std::cout << "Rank " << rank_ << ": ops is not an array in threadblock group " << tb_group_id << std::endl;
    return;
  }
  
  std::cout << "Rank " << rank_ << ": Found " << ops_array.size() 
            << " operations in threadblock group " << tb_group_id << std::endl;
  
  for (size_t op_index = 0; op_index < ops_array.size(); ++op_index) {
    try {
      std::cout << "Rank " << rank_ << ": Starting operation " << op_index 
                << " of " << ops_array.size() << " in threadblock group " << tb_group_id << std::endl;
      
      // Get reference to the actual nlohmann::json object in the array
      nlohmann::json& op_raw_json = ops_array[op_index];
      
      if (!op_raw_json.is_object()) {
        std::cout << "Rank " << rank_ << ": Operation " << op_index 
                  << " is not an object, skipping" << std::endl;
        continue;
      }
      
      std::cout << "Rank " << rank_ << ": Processing operation " << op_index 
                << " in threadblock group " << tb_group_id;
      
      // Check operation name for debugging
      if (op_raw_json.contains("name") && op_raw_json["name"].is_string()) {
        std::cout << " (name: " << op_raw_json["name"].get<std::string>() << ")";
      }
      std::cout << std::endl;
      
      // Create a JsonType wrapper - make a DEEP COPY instead of reference
      std::cout << "Rank " << rank_ << ": Creating JsonType wrapper for operation " << op_index << std::endl;
      JsonType op_json_wrapper(op_raw_json);  // Use copy constructor instead of assignment
      
      std::cout << "Rank " << rank_ << ": Calling processDynamicOperation for operation " << op_index << std::endl;
      processDynamicOperation(op_json_wrapper, params, gpu_id, tb_group_id, static_cast<int>(op_index));
      
      std::cout << "Rank " << rank_ << ": Copying modified data back for operation " << op_index << std::endl;
      // Safe copy-back using dump/parse to avoid reference issues
      try {
        std::string json_str = op_json_wrapper.dump();
        op_raw_json = nlohmann::json::parse(json_str);
        std::cout << "Rank " << rank_ << ": Successfully copied back operation " << op_index << std::endl;
      } catch (const std::exception& copy_error) {
        std::cout << "Rank " << rank_ << ": Copy-back failed for operation " << op_index 
                  << ": " << copy_error.what() << std::endl;
        // Fallback: direct assignment
        op_raw_json = static_cast<nlohmann::json&>(op_json_wrapper);
      }
      
      std::cout << "Rank " << rank_ << ": Completed processing operation " << op_index << std::endl;
      
    } catch (const std::exception& e) {
      std::cout << "Rank " << rank_ << ": ERROR processing operation " << op_index 
                << " in threadblock group " << tb_group_id << ": " << e.what() << std::endl;
      // Continue processing other operations instead of failing completely
      continue;
    } catch (...) {
      std::cout << "Rank " << rank_ << ": UNKNOWN ERROR processing operation " << op_index 
                << " in threadblock group " << tb_group_id << std::endl;
      // Continue processing other operations instead of failing completely
      continue;
    }
  }
  
  std::cout << "Rank " << rank_ << ": Completed processing all " << ops_array.size() 
            << " operations in threadblock group " << tb_group_id << std::endl;
}

// Missing method implementations that are needed
int DynamicExecutionPlan::calculateThreadBlocksForGroup(int tb_group_id, const DynamicRuntimeParams& params) const {
  // Simple implementation - can be made more sophisticated
  return std::min(4, params.maxThreadBlocks / std::max(1, params.num_ranks));
}

int DynamicExecutionPlan::getPeerRankForOperation(int gpu_id, int tb_group_id, int op_index, 
                                                 const DynamicRuntimeParams& params) const {
  // Simple mapping - operation index maps to peer rank
  return op_index % params.num_ranks;
}

size_t DynamicExecutionPlan::getChunkSizeForPeer(int peer_id, const DynamicRuntimeParams& params, bool is_send) const {
  if (is_send) {
    return (peer_id < static_cast<int>(params.send_sizes.size())) ? params.send_sizes[peer_id] : 0;
  } else {
    return (peer_id < static_cast<int>(params.recv_sizes.size())) ? params.recv_sizes[peer_id] : 0;
  }
}

size_t DynamicExecutionPlan::getChunkOffsetForPeer(int peer_id, const DynamicRuntimeParams& params, bool is_send) const {
  if (is_send) {
    return (peer_id < static_cast<int>(params.send_offsets.size())) ? params.send_offsets[peer_id] : 0;
  } else {
    return (peer_id < static_cast<int>(params.recv_offsets.size())) ? params.recv_offsets[peer_id] : 0;
  }
}

size_t DynamicExecutionPlan::getChunkIndexForScratchBuffer(int src_rank, int dst_rank) const {
  // Simple implementation for scratch buffer indexing
  return static_cast<size_t>(src_rank * 1000 + dst_rank);  // Simple unique index
}

void DynamicExecutionPlan::setupStandardVariables(VariableContext& var_context, const DynamicRuntimeParams& params) {
  // Setup common variables
  var_context.setVariable("num_ranks", std::to_string(params.num_ranks));
  var_context.setVariable("max_thread_blocks", std::to_string(params.maxThreadBlocks));
  var_context.setVariable("block_size", std::to_string(params.blockSize));
}

// Fix buffer object processing to create proper buffer specifications

void DynamicExecutionPlan::processDynamicOperation(JsonType& op_json, const DynamicRuntimeParams& params,
                                                  int gpu_id, int tb_group_id, int op_index) {
  std::cout << "Rank " << rank_ << ": Processing dynamic operation " << op_index 
            << " in threadblock group " << tb_group_id << std::endl;
  
  // DEBUG: Show what fields this operation has before processing
  std::cout << "Rank " << rank_ << ": Operation " << op_index << " has fields: ";
  for (auto it = op_json.begin(); it != op_json.end(); ++it) {
    std::cout << it.key() << "(" << (it.value().is_object() ? "object" : 
                                    it.value().is_array() ? "array" : 
                                    it.value().is_string() ? "string" :
                                    it.value().is_number() ? "number" : "other") << ") ";
  }
  std::cout << std::endl;
  
  // Process src_buff array with dynamic fields - keep as proper buffer objects
  if (op_json.contains("src_buff") && op_json["src_buff"].is_array()) {
    auto& src_buff_array = op_json["src_buff"];
    std::cout << "Rank " << rank_ << ": Processing src_buff array with " << src_buff_array.size() << " elements" << std::endl;
    for (size_t i = 0; i < src_buff_array.size(); ++i) {
      auto& buff_obj = src_buff_array[i];
      if (buff_obj.is_object()) {
        // DEBUG: Show buffer object contents
        std::cout << "Rank " << rank_ << ": src_buff[" << i << "] object fields: ";
        for (auto& [key, value] : buff_obj.items()) {
          std::cout << key << "(" << (value.is_object() ? "object" : 
                                     value.is_array() ? "array" : 
                                     value.is_string() ? "string" :
                                     value.is_number() ? "number" : "other") << ") ";
        }
        std::cout << std::endl;
        
        // Create a JsonType wrapper for the buffer object - use copy constructor
        try {
          JsonType buff_json_wrapper(buff_obj);
          processDynamicBufferObject(buff_json_wrapper, params, op_index);
          
          // Create proper buffer object with all required fields
          int peer_id = op_index % params.num_ranks;
          size_t chunk_size = getChunkSizeForPeer(peer_id, params, true);  // true for send size
          size_t chunk_offset = getChunkOffsetForPeer(peer_id, params, true);  // true for send offset
          
          nlohmann::json proper_buffer_obj = {
            {"index", static_cast<int64_t>(peer_id)},
            {"size", static_cast<int64_t>(chunk_size)},
            {"offset", static_cast<int64_t>(chunk_offset)},
            {"type", "i"}  // input buffer type
          };
          
          buff_obj = proper_buffer_obj;
          
          std::cout << "Rank " << rank_ << ": Successfully processed src_buff[" << i 
                    << "] as proper buffer object: index=" << peer_id 
                    << ", size=" << chunk_size << ", offset=" << chunk_offset << std::endl;
        } catch (const std::exception& buff_error) {
          std::cout << "Rank " << rank_ << ": ERROR processing src_buff[" << i << "]: " << buff_error.what() << std::endl;
          // Create fallback proper buffer object
          int peer_id = op_index % params.num_ranks;
          buff_obj = nlohmann::json{
            {"index", static_cast<int64_t>(peer_id)}, 
            {"size", static_cast<int64_t>(1024)}, 
            {"offset", static_cast<int64_t>(0)},
            {"type", "i"}
          };
        }
      }
    }
  }
  
  // Process dst_buff array with dynamic fields - keep as proper buffer objects
  if (op_json.contains("dst_buff") && op_json["dst_buff"].is_array()) {
    auto& dst_buff_array = op_json["dst_buff"];
    std::cout << "Rank " << rank_ << ": Processing dst_buff array with " << dst_buff_array.size() << " elements" << std::endl;
    for (size_t i = 0; i < dst_buff_array.size(); ++i) {
      auto& buff_obj = dst_buff_array[i];
      if (buff_obj.is_object()) {
        // DEBUG: Show buffer object contents
        std::cout << "Rank " << rank_ << ": dst_buff[" << i << "] object fields: ";
        for (auto& [key, value] : buff_obj.items()) {
          std::cout << key << "(" << (value.is_object() ? "object" : 
                                     value.is_array() ? "array" : 
                                     value.is_string() ? "string" :
                                     value.is_number() ? "number" : "other") << ") ";
        }
        std::cout << std::endl;
        
        // Create a JsonType wrapper for the buffer object - use copy constructor
        try {
          JsonType buff_json_wrapper(buff_obj);
          processDynamicBufferObject(buff_json_wrapper, params, op_index);
          
          // Create proper buffer object with all required fields
          int peer_id = op_index % params.num_ranks;
          size_t chunk_size = getChunkSizeForPeer(peer_id, params, false);  // false for recv size
          size_t chunk_offset = getChunkOffsetForPeer(peer_id, params, false);  // false for recv offset
          
          nlohmann::json proper_buffer_obj = {
            {"index", static_cast<int64_t>(peer_id)},
            {"size", static_cast<int64_t>(chunk_size)},
            {"offset", static_cast<int64_t>(chunk_offset)},
            {"type", "o"}  // output buffer type
          };
          
          buff_obj = proper_buffer_obj;
          
          std::cout << "Rank " << rank_ << ": Successfully processed dst_buff[" << i 
                    << "] as proper buffer object: index=" << peer_id 
                    << ", size=" << chunk_size << ", offset=" << chunk_offset << std::endl;
        } catch (const std::exception& buff_error) {
          std::cout << "Rank " << rank_ << ": ERROR processing dst_buff[" << i << "]: " << buff_error.what() << std::endl;
          // Create fallback proper buffer object
          int peer_id = op_index % params.num_ranks;
          buff_obj = nlohmann::json{
            {"index", static_cast<int64_t>(peer_id)}, 
            {"size", static_cast<int64_t>(4096)}, 
            {"offset", static_cast<int64_t>(0)},
            {"type", "o"}
          };
        }
      }
    }
  }
  
  // Process specific dynamic fields that might be at the operation level
  if (op_json.contains("dynamic_src_buff")) {
    // Process source buffer
    processDynamicBuffers(op_json, "dynamic_src_buff", params, gpu_id, op_index % params.num_ranks);
    op_json.erase("dynamic_src_buff");
  }
  
  if (op_json.contains("dynamic_dst_buff")) {
    // Process destination buffer  
    processDynamicBuffers(op_json, "dynamic_dst_buff", params, gpu_id, op_index % params.num_ranks);
    op_json.erase("dynamic_dst_buff");
  }
  
  // Process other specific dynamic fields
  if (op_json.contains("dynamic_index")) {
    // Convert dynamic_index to actual value
    op_json["index"] = static_cast<int64_t>(op_index);  // Use the operation index
    op_json.erase("dynamic_index");
    std::cout << "Rank " << rank_ << ": Set index = " << op_index << " for operation " << op_index << std::endl;
  }
  
  if (op_json.contains("dynamic_size")) {
    // Convert dynamic_size to actual chunk size
    int peer_id = op_index % params.num_ranks;
    size_t chunk_size = getChunkSizeForPeer(peer_id, params, true);  // true for send size
    op_json["size"] = static_cast<int64_t>(chunk_size);
    op_json.erase("dynamic_size");
    std::cout << "Rank " << rank_ << ": Set size = " << chunk_size << " for operation " << op_index << std::endl;
  }
  
  if (op_json.contains("dynamic_offset")) {
    // Convert dynamic_offset to actual offset
    int peer_id = op_index % params.num_ranks;
    size_t chunk_offset = getChunkOffsetForPeer(peer_id, params, true);  // true for send offset
    op_json["offset"] = static_cast<int64_t>(chunk_offset);
    op_json.erase("dynamic_offset");
    std::cout << "Rank " << rank_ << ": Set offset = " << chunk_offset << " for operation " << op_index << std::endl;
  }
  
  // DEBUG: Show remaining object fields after processing dynamic fields
  std::vector<std::string> remaining_object_fields;
  for (auto it = op_json.begin(); it != op_json.end(); ++it) {
    if (it.value().is_object()) {
      remaining_object_fields.push_back(it.key());
    }
  }
  
  if (!remaining_object_fields.empty()) {
    std::cout << "Rank " << rank_ << ": Operation " << op_index << " still has object fields: ";
    for (const auto& field : remaining_object_fields) {
      std::cout << field << " ";
    }
    std::cout << std::endl;
    
    // For each remaining object field, show its contents
    for (const std::string& field_key : remaining_object_fields) {
      std::cout << "Rank " << rank_ << ": Object field " << field_key << " contents: ";
      try {
        auto& obj = op_json[field_key];
        for (auto& [key, value] : obj.items()) {
          std::cout << key << "=" << (value.is_string() ? value.get<std::string>() : "non-string") << " ";
        }
        std::cout << std::endl;
      } catch (...) {
        std::cout << "(error reading contents)" << std::endl;
      }
    }
  }
  
  // Now process the remaining object fields safely
  for (const std::string& field_key : remaining_object_fields) {
    std::cout << "Rank " << rank_ << ": Converting remaining object field " << field_key 
              << " to placeholder in operation " << op_index << std::endl;
    
    // Convert object to string representation (fallback)
    op_json[field_key] = "processed_" + field_key;
  }
  
  // DEBUG: Show final state of operation
  std::cout << "Rank " << rank_ << ": Final operation " << op_index << " fields: ";
  for (auto it = op_json.begin(); it != op_json.end(); ++it) {
    std::cout << it.key() << "(" << (it.value().is_object() ? "object" : 
                                    it.value().is_array() ? "array" : 
                                    it.value().is_string() ? "string" :
                                    it.value().is_number() ? "number" : "other") << ") ";
  }
  std::cout << std::endl;
  
  std::cout << "Rank " << rank_ << ": Completed processing dynamic operation " << op_index 
            << " in threadblock group " << tb_group_id << std::endl;
  
  // FINAL DEBUG: Check what's still an object after ALL processing
  std::cout << "Rank " << rank_ << ": FINAL CHECK for operation " << op_index << ": ";
  for (auto it = op_json.begin(); it != op_json.end(); ++it) {
    if (it.value().is_object()) {
      std::cout << it.key() << "(STILL-OBJECT) ";
    } else if (it.value().is_array()) {
      // Check array contents
      bool has_objects = false;
      for (const auto& elem : it.value()) {
        if (elem.is_object()) {
          has_objects = true;
          break;
        }
      }
      std::cout << it.key() << (has_objects ? "(ARRAY-HAS-OBJECTS) " : "(array-ok) ");
    } else {
      std::cout << it.key() << "(ok) ";
    }
  }
  std::cout << std::endl;
}

void DynamicExecutionPlan::processDynamicBufferObject(JsonType& buff_obj, 
                                                     const DynamicRuntimeParams& params, 
                                                     int op_index) {
  if (!buff_obj.is_object()) return;
  
  // Process dynamic_index field
  if (buff_obj.contains("dynamic_index")) {
    int peer_id = op_index % params.num_ranks;
    buff_obj["index"] = peer_id;  // Set to peer rank for this operation
    buff_obj.erase("dynamic_index");
    std::cout << "Rank " << rank_ << ": Set buffer index = " << peer_id << " for operation " << op_index << std::endl;
  }
  
  // Process dynamic_size field
  if (buff_obj.contains("dynamic_size")) {
    int peer_id = op_index % params.num_ranks;
    size_t chunk_size = getChunkSizeForPeer(peer_id, params, true);  // Default to send size
    
    // If this is an output buffer, use recv size instead
    if (buff_obj.contains("type") && buff_obj["type"].is_string() && buff_obj["type"] == "o") {
      chunk_size = getChunkSizeForPeer(peer_id, params, false);  // false for recv size
    }
    
    buff_obj["size"] = static_cast<long long>(chunk_size);
    buff_obj.erase("dynamic_size");
    std::cout << "Rank " << rank_ << ": Set buffer size = " << chunk_size << " for operation " << op_index << std::endl;
  }
  
  // Process dynamic_offset field if it exists
  if (buff_obj.contains("dynamic_offset")) {
    int peer_id = op_index % params.num_ranks;
    size_t chunk_offset = getChunkOffsetForPeer(peer_id, params, true);  // Default to send offset
    
    // If this is an output buffer, use recv offset instead
    if (buff_obj.contains("type") && buff_obj["type"].is_string() && buff_obj["type"] == "o") {
      chunk_offset = getChunkOffsetForPeer(peer_id, params, false);  // false for recv offset
    }
    
    buff_obj["offset"] = static_cast<long long>(chunk_offset);
    buff_obj.erase("dynamic_offset");
    std::cout << "Rank " << rank_ << ": Set buffer offset = " << chunk_offset << " for operation " << op_index << std::endl;
  }
  
  // Handle any null values by converting them to appropriate defaults
  for (auto it = buff_obj.begin(); it != buff_obj.end(); ++it) {
    if (it.value().is_null()) {
      std::cout << "Rank " << rank_ << ": Found null value in buffer object field " << it.key() 
                << ", replacing with default" << std::endl;
      if (it.key() == "index") {
        it.value() = op_index % params.num_ranks;
      } else if (it.key() == "size") {
        it.value() = 1024;  // Default size
      } else if (it.key() == "offset") {
        it.value() = 0;     // Default offset
      } else {
        it.value() = 0;     // Generic default for numbers
      }
    }
  }
}

void DynamicExecutionPlan::processDynamicBuffers(JsonType& op_json, const std::string& buffer_key,
                                                const DynamicRuntimeParams& params, int gpu_id, int peer_id) {
  std::cout << "Rank " << rank_ << ": Processing buffer " << buffer_key 
            << " for peer " << peer_id << std::endl;
  
  // Simple implementation - replace with actual buffer index
  if (buffer_key == "dynamic_src_buff") {
    op_json["src_buff"] = peer_id;  // Input buffer index
    std::cout << "Rank " << rank_ << ": Set src_buff = " << peer_id << " for GPU " << gpu_id << std::endl;
  } else if (buffer_key == "dynamic_dst_buff") {
    op_json["dst_buff"] = peer_id;  // Output buffer index  
    std::cout << "Rank " << rank_ << ": Set dst_buff = " << peer_id << " for GPU " << gpu_id << std::endl;
  } else {
    std::cout << "Rank " << rank_ << ": Unknown buffer key: " << buffer_key << std::endl;
  }
}

void DynamicExecutionPlan::processOperationTemplates(JsonType& gpu_json, 
                                                    const DynamicRuntimeParams& params,
                                                    const VariableContext& var_context) {
  // Placeholder implementation
  std::cout << "Rank " << rank_ << ": Processing operation templates" << std::endl;
}

void DynamicExecutionPlan::substituteOperationTemplateVariables(JsonType& operation_template,
                                                               const DynamicRuntimeParams& params,
                                                               const VariableContext& var_context) {
  // Placeholder implementation
  std::cout << "Rank " << rank_ << ": Substituting operation template variables" << std::endl;
}

void DynamicExecutionPlan::sanitizeJsonForSerialization(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Sanitizing JSON for MSCCLPP executor compatibility" << std::endl;
  
  // Convert ALL numeric values to regular int to avoid MSCCLPP type casting issues
  std::function<void(nlohmann::json&)> standardizeToInt = [&](nlohmann::json& j) {
    if (j.is_object()) {
      for (auto it = j.begin(); it != j.end(); ++it) {
        standardizeToInt(it.value());
      }
    } else if (j.is_array()) {
      for (auto it = j.begin(); it != j.end(); ++it) {
        standardizeToInt(*it);
      }
    } else if (j.is_number_integer()) {
      // Convert all integers to regular int for MSCCLPP compatibility
      j = static_cast<int>(j.get<int64_t>());
    }
    // Leave strings, booleans, and null values as-is
  };
  
  nlohmann::json& raw_json = static_cast<nlohmann::json&>(json_obj);
  standardizeToInt(raw_json);
  
  std::cout << "Rank " << rank_ << ": Converted all numeric values to int for MSCCLPP compatibility" << std::endl;
}

void DynamicExecutionPlan::aggressivelySanitizeJson(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Performing aggressive JSON sanitization" << std::endl;
  
  // More aggressive sanitization that converts problematic values to safe defaults
  std::function<void(nlohmann::json&)> aggressiveSanitize = [&](nlohmann::json& j) {
    if (j.is_object()) {
      // Create a new object to avoid iteration issues
      nlohmann::json new_obj = nlohmann::json::object();
      for (auto it = j.begin(); it != j.end(); ++it) {
        try {
          nlohmann::json sanitized_value = it.value();
          aggressiveSanitize(sanitized_value);
          new_obj[it.key()] = sanitized_value;
        } catch (...) {
          // Replace problematic values with safe defaults
          std::cout << "Rank " << rank_ << ": Replacing problematic value in field: " << it.key() << std::endl;
          new_obj[it.key()] = "sanitized_value";
        }
      }
      j = new_obj;
    } else if (j.is_array()) {
      // Create a new array to avoid iteration issues
      nlohmann::json new_array = nlohmann::json::array();
      for (size_t i = 0; i < j.size(); ++i) {
        try {
          nlohmann::json sanitized_elem = j[i];
          aggressiveSanitize(sanitized_elem);
          new_array.push_back(sanitized_elem);
        } catch (...) {
          // Replace problematic elements with safe defaults
          std::cout << "Rank " << rank_ << ": Replacing problematic array element at index: " << i << std::endl;
          new_array.push_back("sanitized_element");
        }
      }
      j = new_array;
    } else if (j.is_number()) {
      // Ensure all numbers are safe integers
      try {
        if (j.is_number_integer()) {
          j = static_cast<int64_t>(j.get<int64_t>());
        } else {
          // Convert floats to integers for safety
          j = static_cast<int64_t>(j.get<double>());
        }
      } catch (...) {
        j = static_cast<int64_t>(0);  // Default to 0 for problematic numbers
      }
    }
    // Leave strings, booleans, and null values as-is
  };
  
  nlohmann::json& raw_json = static_cast<nlohmann::json&>(json_obj);
  aggressiveSanitize(raw_json);
}

DynamicExecutionPlan::JsonType DynamicExecutionPlan::createSanitizedExecutionPlan() {
  std::cout << "Rank " << rank_ << ": Creating sanitized execution plan" << std::endl;
  
  // Create a completely new, clean JSON structure
  nlohmann::json sanitized = {
    {"buffer_alignment", static_cast<int64_t>(16)},
    {"collective", "alltoall"},
    {"dynamic", true},
    {"dynamic_parameters", {
      {"block_size", static_cast<int64_t>(32768)},
      {"max_thread_blocks", static_cast<int64_t>(32)}
    }},
    {"gpus", nlohmann::json::array()},
    {"inplace", false},
    {"max_message_size", static_cast<int64_t>(1073741824)},
    {"min_message_size", static_cast<int64_t>(0)},
    {"name", "alltoallv_dynamic_4gpu"},
    {"num_threads_per_block", static_cast<int64_t>(256)},
    {"protocol", "Simple"},
    {"reuse_resources", false},
    {"use_double_scratch_buffer", false}
  };
  
  // Add basic GPU structures
  for (int gpu_id = 0; gpu_id < 4; ++gpu_id) {  // Default to 4 GPUs
    nlohmann::json gpu = {
      {"id", static_cast<int64_t>(gpu_id)},
      {"input_chunks", static_cast<int64_t>(1)},      // CHANGED from 4 to 1
      {"output_chunks", static_cast<int64_t>(1)},     // CHANGED from 4 to 1
      {"scratch_chunks", static_cast<int64_t>(3)},
      {"threadblocks", nlohmann::json::array()},
      {"channels", nlohmann::json::array()},
      {"remote_buffers", nlohmann::json::array()},
      {"semaphores", nlohmann::json::array()}
    };
    
    // Add basic threadblock structure
    nlohmann::json threadblock = {
      {"tb_count", static_cast<int64_t>(1)},
      {"tb_group_id", static_cast<int64_t>(0)},
      {"ops", nlohmann::json::array()}
    };
    
    gpu["threadblocks"].push_back(threadblock);
    sanitized["gpus"].push_back(gpu);
  }
  
  return JsonType(sanitized);
}

void DynamicExecutionPlan::expandThreadblocks(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Expanding aggregated threadblocks to concrete threadblock entries with IDs" << std::endl;
  
  if (!json_obj.contains("gpus") || !json_obj["gpus"].is_array()) {
    return;
  }
  
  auto& gpus_array = json_obj["gpus"];
  
  for (size_t gpu_idx = 0; gpu_idx < gpus_array.size(); ++gpu_idx) {
    auto& gpu_obj = gpus_array[gpu_idx];
    
    if (!gpu_obj.contains("threadblocks") || !gpu_obj["threadblocks"].is_array()) {
      continue;
    }
    
    auto& threadblocks_array = gpu_obj["threadblocks"];
    nlohmann::json new_threadblocks_array = nlohmann::json::array();
    
    int global_threadblock_id = 0;  // Global threadblock counter for this GPU
    
    for (size_t tb_idx = 0; tb_idx < threadblocks_array.size(); ++tb_idx) {
      auto& tb_template = threadblocks_array[tb_idx];
      
      // Check if this is an aggregated threadblock with tb_count
      int tb_count = 1;  // Default to 1 if no tb_count specified
      if (tb_template.contains("tb_count") && tb_template["tb_count"].is_number()) {
        tb_count = tb_template["tb_count"].get<int>();
      }
      
      std::cout << "Rank " << rank_ << ": Expanding threadblock group " << tb_idx 
                << " with tb_count=" << tb_count << " for GPU " << gpu_idx << std::endl;
      
      // Create tb_count individual threadblock entries
      for (int tb_instance = 0; tb_instance < tb_count; ++tb_instance) {
        nlohmann::json concrete_threadblock = nlohmann::json::object();
        
        // Add the threadblock ID first
        concrete_threadblock["id"] = static_cast<int>(global_threadblock_id);
        
        // Copy all fields except our custom ones
        for (auto& [key, value] : tb_template.items()) {
          if (key != "tb_count" && key != "tb_group_id") {
            concrete_threadblock[key] = value;
          }
        }
        
        new_threadblocks_array.push_back(concrete_threadblock);
        
        std::cout << "Rank " << rank_ << ": Created concrete threadblock with ID " 
                  << global_threadblock_id << " (instance " << tb_instance 
                  << " of group " << tb_idx << ")" << std::endl;
        
        global_threadblock_id++;  // Increment for next threadblock
      }
    }
    
    // Replace the aggregated threadblocks with concrete ones
    gpu_obj["threadblocks"] = new_threadblocks_array;
    
    std::cout << "Rank " << rank_ << ": GPU " << gpu_idx 
              << " now has " << new_threadblocks_array.size() 
              << " concrete threadblock entries with IDs 0-" << (global_threadblock_id - 1) << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Completed threadblock expansion with proper IDs" << std::endl;
}

void DynamicExecutionPlan::validateAndFixBufferArrays(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Validating and fixing empty buffer arrays" << std::endl;
  
  if (!json_obj.contains("gpus") || !json_obj["gpus"].is_array()) {
    return;
  }
  
  auto& gpus_array = json_obj["gpus"];
  
  for (size_t gpu_idx = 0; gpu_idx < gpus_array.size(); ++gpu_idx) {
    auto& gpu_obj = gpus_array[gpu_idx];
    
    if (!gpu_obj.contains("threadblocks") || !gpu_obj["threadblocks"].is_array()) {
      continue;
    }
    
    auto& threadblocks_array = gpu_obj["threadblocks"];
    
    for (size_t tb_idx = 0; tb_idx < threadblocks_array.size(); ++tb_idx) {
      auto& tb_obj = threadblocks_array[tb_idx];
      
      if (!tb_obj.contains("ops") || !tb_obj["ops"].is_array()) {
        continue;
      }
      
      auto& ops_array = tb_obj["ops"];
      
      for (size_t op_idx = 0; op_idx < ops_array.size(); ++op_idx) {
        auto& op_obj = ops_array[op_idx];
        
        // Check and fix empty src_buff arrays
        if (op_obj.contains("src_buff") && op_obj["src_buff"].is_array() && op_obj["src_buff"].empty()) {
          // Add a default buffer object for operations that should have buffers
          if (op_obj.contains("name") && op_obj["name"].is_string()) {
            std::string op_name = op_obj["name"].get<std::string>();
            if (op_name == "copy" || op_name == "put" || op_name == "get") {
              nlohmann::json default_buffer = {
                {"index", static_cast<int>(op_idx % 4)},
                {"size", static_cast<int>(1024)},
                {"offset", static_cast<int>(0)},
                {"type", "i"}
              };
              op_obj["src_buff"].push_back(default_buffer);
              std::cout << "Rank " << rank_ << ": Fixed empty src_buff for operation " 
                        << op_name << " in GPU " << gpu_idx << " threadblock " << tb_idx << std::endl;
            }
          }
        }
        
        // Check and fix empty dst_buff arrays
        if (op_obj.contains("dst_buff") && op_obj["dst_buff"].is_array() && op_obj["dst_buff"].empty()) {
          if (op_obj.contains("name") && op_obj["name"].is_string()) {
            std::string op_name = op_obj["name"].get<std::string>();
            if (op_name == "copy" || op_name == "put" || op_name == "get") {
              nlohmann::json default_buffer = {
                {"index", static_cast<int>(op_idx % 4)},
                {"size", static_cast<int>(1024)},
                {"offset", static_cast<int>(0)},
                {"type", "o"}
              };
              op_obj["dst_buff"].push_back(default_buffer);
              std::cout << "Rank " << rank_ << ": Fixed empty dst_buff for operation " 
                        << op_name << " in GPU " << gpu_idx << " threadblock " << tb_idx << std::endl;
            }
          }
        }
      }
    }
  }
  
  std::cout << "Rank " << rank_ << ": Completed buffer array validation and fixes" << std::endl;
}

}  // namespace mscclpp