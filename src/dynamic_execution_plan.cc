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
    
    // Debug: Verify input buffer has data before execution
    std::vector<uint8_t> host_send_debug(1000);
    cudaMemcpy(host_send_debug.data(), sendBuffer, 1000, cudaMemcpyDeviceToHost);
    std::cout << "Rank " << allToAllv->rank_ << ": Input buffer before execution (first 10 bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(host_send_debug[i]) << " ";
    }
    std::cout << std::endl;
    
    // Execute the operation
    allToAllv->execute(
      sendBuffer, sendSizes, params.send_offsets,
      recvBuffer, recvSizes, params.recv_offsets,
      comm, executor, stream);
    
    // Ensure execution completes
    cudaStreamSynchronize(stream);
    
    // Debug: Check if anything was written to output buffer
    std::vector<uint8_t> host_recv_debug(1000);
    cudaMemcpy(host_recv_debug.data(), recvBuffer, 1000, cudaMemcpyDeviceToHost);
    std::cout << "Rank " << allToAllv->rank_ << ": Output buffer after execution (first 10 bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(host_recv_debug[i]) << " ";
    }
    std::cout << std::endl;
    
    // Synchronize and cleanup
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
    
    std::cout << "Rank " << rank_ << ": Executing with total send size: " 
              << totalSendSize << ", total recv size: " << totalRecvSize << std::endl;
    
    // Step 3: Execute the concrete plan using the MSCCLPP executor
    // MSCCLPP expects buffer sizes as element counts when using UINT32 DataType
    size_t sendElementCount = totalSendSize / sizeof(uint32_t);
    size_t recvElementCount = totalRecvSize / sizeof(uint32_t);
    
    // Debug: Verify input buffer has data before execution
    std::vector<uint8_t> host_send_debug(1000);
    cudaMemcpy(host_send_debug.data(), send_buff, 1000, cudaMemcpyDeviceToHost);
    std::cout << "Rank " << rank_ << ": Input buffer before execution (first 10 bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(host_send_debug[i]) << " ";
    }
    std::cout << std::endl;
    
    // Execute with MSCCLPP
    executor->execute(
      rank_,                      
      send_buff,                  
      recv_buff,                  
      sendElementCount,           
      recvElementCount,           
      mscclpp::DataType::UINT32,  
      *executionPlan,             
      stream,                     
      mscclpp::PacketType::LL16   
    );
    
    // Ensure execution completes
    cudaStreamSynchronize(stream);
    
    // Debug: Check if anything was written to output buffer
    std::vector<uint8_t> host_recv_debug(1000);
    cudaMemcpy(host_recv_debug.data(), recv_buff, 1000, cudaMemcpyDeviceToHost);
    std::cout << "Rank " << rank_ << ": Output buffer after execution (first 10 bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(host_recv_debug[i]) << " ";
    }
    std::cout << std::endl;
    
    // NEW: Debug note about scratch buffer configuration
    // MSCCLPP internally manages scratch buffers based on the execution plan
    std::cout << "Rank " << rank_ << ": NOTE: MSCCLPP manages scratch buffers internally based on execution plan configuration" << std::endl;
    std::cout << "Rank " << rank_ << ": Checking scratch buffer configuration in concrete plan..." << std::endl;
    
    // Check the concrete plan we just created to verify scratch_chunks
    // Use the public getter instead of accessing private member directly
    const std::string& tempFilePath = plan_.getTempFilePath();
    if (!tempFilePath.empty()) {
      try {
        std::ifstream planFile(tempFilePath);
        if (planFile.is_open()) {
          nlohmann::json planObj;
          planFile >> planObj;
          planFile.close();
          
          // Check scratch buffer configuration for this rank's GPU
          bool foundGpu = false;
          if (planObj.contains("gpus") && planObj["gpus"].is_array()) {
            for (const auto& gpu : planObj["gpus"]) {
              if (gpu.contains("id") && gpu["id"].get<int>() == rank_) {
                foundGpu = true;
                int scratchChunks = gpu.value("scratch_chunks", 0);
                std::cout << "Rank " << rank_ << ": GPU " << rank_ << " has scratch_chunks = " << scratchChunks << std::endl;
                if (scratchChunks == 0) {
                  std::cout << "Rank " << rank_ << ": WARNING: scratch_chunks is 0! Data transfer will fail!" << std::endl;
                  std::cout << "Rank " << rank_ << ": The execution plan must have scratch_chunks > 0 for AllToAllV!" << std::endl;
                } else {
                  std::cout << "Rank " << rank_ << ": Good: scratch_chunks = " << scratchChunks << " (should enable data transfer)" << std::endl;
                }
                break;
              }
            }
            if (!foundGpu) {
              std::cout << "Rank " << rank_ << ": WARNING: Could not find GPU " << rank_ << " in execution plan!" << std::endl;
            }
          }
        }
      } catch (const std::exception& e) {
        std::cout << "Rank " << rank_ << ": Could not verify scratch buffer configuration: " << e.what() << std::endl;
      }
    }
    
    // Synchronize and cleanup
    cudaStreamDestroy(stream);
    
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
  std::cout << "Rank " << rank_ << ": Starting enhanced template instantiation..." << std::endl;

  try {
    // Make a deep copy of the original template
    JsonType workingJson = *templateJson_;  // Copy the template
    
    // STEP 1: Process dynamic template fields FIRST (but preserve dynamic_tbgroup_id)
    std::cout << "Rank " << rank_ << ": Processing dynamic template fields..." << std::endl;
    processDynamicTemplate(workingJson, params);
    
    // STEP 2: Expand threadblocks AFTER template processing
    std::cout << "Rank " << rank_ << ": Expanding threadblocks after template processing..." << std::endl;
    expandThreadblocks(workingJson);
    
    // STEP 3: Final sanitization
    std::cout << "Rank " << rank_ << ": Applying final sanitization..." << std::endl;
    sanitizeJsonForSerialization(workingJson);
    
    // STEP 4: CRITICAL - Final enforcement of AllToAllV settings
    std::cout << "Rank " << rank_ << ": Final enforcement of AllToAllV settings..." << std::endl;
    enforceAllToAllVSettings(workingJson, params);
    
    std::cout << "Rank " << rank_ << ": Template instantiation completed successfully" << std::endl;
    
    // Generate the final JSON string
    std::string concretePlan = static_cast<nlohmann::json&>(workingJson).dump(2);  // Pretty print with 2-space indent
    
    // DEBUG: Save the concrete execution plan to build directory for inspection
    try {
      // Create a unique filename based on rank and timestamp
      auto now = std::chrono::steady_clock::now().time_since_epoch().count();
      std::string debugFileName = "./concrete_alltoallv_plan_" + std::to_string(rank_) + 
                                  "_" + std::to_string(now) + ".json";
      
      std::ofstream debugFile(debugFileName);
      if (debugFile.is_open()) {
        debugFile << concretePlan;
        debugFile.close();
        std::cout << "Rank " << rank_ << ": DEBUG: Saved concrete execution plan to " << debugFileName << std::endl;
        std::cout << "Rank " << rank_ << ": DEBUG: Plan size: " << concretePlan.length() << " characters" << std::endl;
        
        // Print a summary of the concrete plan structure
        nlohmann::json planJson = nlohmann::json::parse(concretePlan);
        if (planJson.contains("gpus") && planJson["gpus"].is_array()) {
          std::cout << "Rank " << rank_ << ": DEBUG: Concrete plan contains " << planJson["gpus"].size() << " GPUs" << std::endl;
          
          for (size_t gpu_idx = 0; gpu_idx < planJson["gpus"].size(); ++gpu_idx) {
            auto& gpu = planJson["gpus"][gpu_idx];
            if (gpu.contains("threadblocks") && gpu["threadblocks"].is_array()) {
              std::cout << "Rank " << rank_ << ": DEBUG: GPU " << gpu_idx << " has " 
                        << gpu["threadblocks"].size() << " concrete threadblocks" << std::endl;
              
              // Count operations per GPU
              int total_ops = 0;
              for (const auto& tb : gpu["threadblocks"]) {
                if (tb.contains("ops") && tb["ops"].is_array()) {
                  total_ops += tb["ops"].size();
                }
              }
              std::cout << "Rank " << rank_ << ": DEBUG: GPU " << gpu_idx << " has total " 
                        << total_ops << " operations across all threadblocks" << std::endl;
            }
          }
        }
        
      } else {
        std::cout << "Rank " << rank_ << ": WARNING: Could not create debug file " << debugFileName << std::endl;
      }
      
    } catch (const std::exception& debug_e) {
      std::cout << "Rank " << rank_ << ": WARNING: Failed to save debug concrete plan: " 
                << debug_e.what() << std::endl;
      // Don't fail the entire function just because debug save failed
    }
    
    // Return the JSON as a string
    return concretePlan;
    
  } catch (const std::exception& e) {
    std::cout << "Rank " << rank_ << ": Error during template instantiation: " << e.what() << std::endl;
    throw;
  }
}

// Add this new method
void DynamicExecutionPlan::enforceAllToAllVSettings(JsonType& json_obj, const DynamicRuntimeParams& params) {
  std::cout << "Rank " << rank_ << ": Enforcing final AllToAllV settings..." << std::endl;
  
  // Fix collective type
  json_obj["collective"] = "alltoallv";
  std::cout << "Rank " << rank_ << ": Set collective = alltoallv" << std::endl;
  
  // Remove template-specific fields from concrete plan
  if (json_obj.contains("dynamic")) {
    json_obj.erase("dynamic");
    std::cout << "Rank " << rank_ << ": Removed 'dynamic' field from concrete plan" << std::endl;
  }
  
  if (json_obj.contains("dynamic_parameters")) {
    json_obj.erase("dynamic_parameters");
    std::cout << "Rank " << rank_ << ": Removed 'dynamic_parameters' field from concrete plan" << std::endl;
  }
  
  // Also remove any other template-specific fields that shouldn't be in concrete plans
  if (json_obj.contains("is_dynamic")) {
    json_obj.erase("is_dynamic");
    std::cout << "Rank " << rank_ << ": Removed 'is_dynamic' field from concrete plan" << std::endl;
  }
  
  // Process each GPU and force correct settings
  if (json_obj.contains("gpus") && json_obj["gpus"].is_array()) {
    auto& gpus_array = json_obj["gpus"];
    for (size_t gpu_idx = 0; gpu_idx < gpus_array.size(); ++gpu_idx) {
      auto& gpu_json = gpus_array[gpu_idx];
      
      // FORCE chunk counts to 1 for AllToAllV
      gpu_json["input_chunks"] = 1;
      gpu_json["output_chunks"] = 1;
      
      std::cout << "Rank " << rank_ << ": FORCED GPU " << gpu_idx 
                << " input_chunks=1, output_chunks=1" << std::endl;
    }
  }
  
  std::cout << "Rank " << rank_ << ": Completed AllToAllV settings enforcement and template cleanup" << std::endl;
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
  
  // CRITICAL FIX: Always set scratch_chunks for AllToAllV
  // Each GPU needs scratch buffers to receive data from other ranks
  int scratch_chunks = params.num_ranks - 1;  // One scratch chunk per remote peer
  gpu_json["scratch_chunks"] = scratch_chunks;
  std::cout << "Rank " << rank_ << ": FORCED scratch_chunks = " << scratch_chunks 
            << " for GPU " << gpu_id << " (AllToAllV requirement)" << std::endl;
  
  // Remove any dynamic scratch chunks field
  if (gpu_json.contains("dynamic_scratch_chunks")) {
    gpu_json.erase("dynamic_scratch_chunks");
  }
  
  // CRITICAL: Force input_chunks and output_chunks to 1 for alltoallv
  // This must come after all other processing to ensure it's not overwritten
  gpu_json["input_chunks"] = 1;
  gpu_json["output_chunks"] = 1;
  std::cout << "Rank " << rank_ << ": FORCED input_chunks = 1, output_chunks = 1 for alltoallv compatibility" << std::endl;
  
  // Ensure proper type for existing fields to avoid number/number type conflicts
  if (gpu_json.contains("input_chunks") && !gpu_json["input_chunks"].is_number_integer()) {
    gpu_json["input_chunks"] = 1;  // Force to 1
  }
  if (gpu_json.contains("output_chunks") && !gpu_json["output_chunks"].is_number_integer()) {
    gpu_json["output_chunks"] = 1;  // Force to 1
  }
  if (gpu_json.contains("scratch_chunks") && !gpu_json["scratch_chunks"].is_number_integer()) {
    gpu_json["scratch_chunks"] = scratch_chunks;
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
  if (!gpu_json.contains("threadblocks") || !gpu_json["threadblocks"].is_array()) {
    std::cout << "Rank " << rank_ << ": No threadblocks array found for GPU " << gpu_id << std::endl;
    return;
  }

  auto& threadblocks_array = gpu_json["threadblocks"];
  
  for (size_t i = 0; i < threadblocks_array.size(); ++i) {
    auto& tb_raw_json = threadblocks_array[i];
    
    if (!tb_raw_json.is_object()) {
      std::cout << "Rank " << rank_ << ": Threadblock " << i << " is not an object, skipping" << std::endl;
      continue;
    }

    try {
      JsonType tb_json_wrapper(tb_raw_json);
      
      if (tb_json_wrapper.contains("dynamic_tbgroup_id")) {
        int tb_group_id = tb_json_wrapper.value("dynamic_tbgroup_id", -1);
        std::cout << "Rank " << rank_ << ": Processing dynamic threadblock group " << tb_group_id 
                  << " (index " << i << ")" << std::endl;
        
        processDynamicThreadblock(tb_json_wrapper, params, gpu_id, tb_group_id);
        
        // CRITICAL CHANGE: DO NOT remove dynamic_tbgroup_id here
        // Let expandThreadblocks() handle the removal after expansion
        // tb_json_wrapper.erase("dynamic_tbgroup_id");  // COMMENT OUT THIS LINE
        
      } else {
        std::cout << "Rank " << rank_ << ": Threadblock " << i 
                  << " does not have dynamic_tbgroup_id, processing as static" << std::endl;
        
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
      
    } catch (const std::exception& tb_error) {
      std::cout << "Rank " << rank_ << ": ERROR processing threadblock at index " << i 
                << ": " << tb_error.what() << std::endl;
      // Continue processing other threadblocks
      continue;
    } catch (...) {
      std::cout << "Rank " << rank_ << ": UNKNOWN ERROR processing threadblock at index " << i << std::endl;
      // Continue processing other threadblocks
      continue;
    }
  }
  
  std::cout << "Rank " << rank_ << ": Completed processing all " << threadblocks_array.size() 
            << " threadblocks for GPU " << gpu_id << std::endl;
}

void DynamicExecutionPlan::processDynamicThreadblock(JsonType& tb_json, const DynamicRuntimeParams& params, 
                                                    int gpu_id, int tb_group_id) {
  std::cout << "Rank " << rank_ << ": Processing threadblock id=" << tb_group_id 
            << " for GPU " << gpu_id << std::endl;
  
  // CRITICAL FIX: tb_group_id here is the original template dynamic_tbgroup_id (0, 1, 2)
  // NOT the expanded concrete threadblock id. It directly maps to peer groups.
  
  int original_peer_group = tb_group_id;  // Direct mapping: 0→peer0, 1→peer1, 2→peer2
  
  // Calculate actual target peer (skipping self)
  int target_peer;
  if (original_peer_group < rank_) {
    target_peer = original_peer_group;      // peer_group < rank: target directly
  } else {
    target_peer = original_peer_group + 1;  // peer_group >= rank: skip self, +1
  }
  
  // Ensure we don't go out of bounds
  if (target_peer >= params.num_ranks) {
    std::cout << "Rank " << rank_ << ": ERROR: target_peer " << target_peer 
              << " exceeds num_ranks " << params.num_ranks << std::endl;
    return;
  }
  
  std::cout << "Rank " << rank_ << ": Threadblock " << tb_group_id 
            << " (template_peer_group=" << original_peer_group 
            << ") → target_peer=" << target_peer << std::endl;
  
  // Process operations within this threadblock, passing the correct target peer
  if (tb_json.contains("ops")) {
    auto& ops_array = tb_json["ops"];
    if (ops_array.is_array()) {
      for (size_t op_idx = 0; op_idx < ops_array.size(); ++op_idx) {
        auto& op_json = ops_array[op_idx];
        if (op_json.is_object()) {
          JsonType op_wrapper(op_json);
          processDynamicOperation(op_wrapper, params, gpu_id, target_peer, static_cast<int>(op_idx));
          op_json = static_cast<nlohmann::json&>(op_wrapper);
        }
      }
    }
  }
  
  std::cout << "Rank " << rank_ << ": Completed threadblock processing for id=" << tb_group_id << std::endl;
}

void DynamicExecutionPlan::processDynamicOperation(JsonType& op_json, const DynamicRuntimeParams& params,
                                                  int gpu_id, int target_peer, int op_index) {
  std::cout << "Rank " << rank_ << ": Processing dynamic operation " << op_index 
            << " targeting peer " << target_peer << std::endl;
  
  // Get operation name to determine processing strategy
  std::string op_name = op_json.value("name", "unknown");
  
  // Skip operations that don't need buffer processing
  if (op_name == "nop" || op_name == "signal" || op_name == "wait") {
    std::cout << "Rank " << rank_ << ": Skipping buffer processing for " << op_name << " operation" << std::endl;
    return;
  }
  
  std::cout << "Rank " << rank_ << ": Operation " << op_index << " (" << op_name 
            << ") → target_peer=" << target_peer << std::endl;

  // CRITICAL FIX: Calculate correct data sizes for AllToAllV
  // In AllToAllV: rank rank_ sends (rank_ + 1) * 1024 bytes to ALL peers
  size_t send_size_per_peer = (rank_ + 1) * 1024;  // Use rank_ instead of gpu_id
  size_t receive_size_from_peer = (target_peer + 1) * 1024;
  
  std::cout << "Rank " << rank_ << ": AllToAllV sizing - send_per_peer=" << send_size_per_peer 
            << ", receive_from_" << target_peer << "=" << receive_size_from_peer << std::endl;

  // Process src_buff array - COMPLETELY REPLACE with correct values
  if (op_json.contains("src_buff") && op_json["src_buff"].is_array()) {
    auto& src_buff_array = op_json["src_buff"];
    std::cout << "Rank " << rank_ << ": Processing src_buff array with " << src_buff_array.size() << " elements" << std::endl;
    for (size_t i = 0; i < src_buff_array.size(); ++i) {
      auto& buff_obj = src_buff_array[i];
      if (buff_obj.is_object()) {
        nlohmann::json proper_buffer_obj;
        
        // Check if this is a COPY operation that should read from scratch buffer
        if (op_name == "copy" && buff_obj.value("type", "") == "s") {
          // COPY from scratch buffer (data received from remote ranks)
          size_t recv_size_elements = receive_size_from_peer / sizeof(uint32_t);
          size_t recv_offset = 0;
          // Calculate offset for data received from target_peer
          for (int sender = 0; sender < target_peer; ++sender) {
            recv_offset += (sender + 1) * 1024;
          }
          size_t recv_offset_elements = recv_offset / sizeof(uint32_t);
          
          proper_buffer_obj = {
            {"index", static_cast<uint32_t>(1)},  // Scratch buffer uses index 1
            {"size", static_cast<uint32_t>(recv_size_elements)},
            {"offset", static_cast<uint32_t>(recv_offset_elements)},
            {"type", "s"}  // Reading from scratch buffer
          };
        } else {
          // Normal case: read from input buffer
          size_t src_offset_bytes = target_peer * send_size_per_peer;
          size_t send_size_elements = send_size_per_peer / sizeof(uint32_t);
          size_t src_offset_elements = src_offset_bytes / sizeof(uint32_t);
          
          proper_buffer_obj = {
            {"index", static_cast<uint32_t>(0)},  // Input buffer uses index 0
            {"size", static_cast<uint32_t>(send_size_elements)},
            {"offset", static_cast<uint32_t>(src_offset_elements)},
            {"type", "i"}  // Input buffer
          };
        }
        
        buff_obj = proper_buffer_obj;
        
        std::cout << "Rank " << rank_ << ": CREATED src_buff[" << i << "] = " 
                  << buff_obj.dump() << std::endl;
      }
    }
  }

  // Process dst_buff array - COMPLETELY REPLACE with correct values
  if (op_json.contains("dst_buff") && op_json["dst_buff"].is_array()) {
    auto& dst_buff_array = op_json["dst_buff"];
    std::cout << "Rank " << rank_ << ": Processing dst_buff array with " << dst_buff_array.size() << " elements" << std::endl;
    for (size_t i = 0; i < dst_buff_array.size(); ++i) {
      auto& buff_obj = dst_buff_array[i];
      if (buff_obj.is_object()) {
        size_t dst_size, dst_offset;
        int dst_index;
        
        if (op_name == "copy") {
          // CRITICAL FIX: For COPY operations, use CONSISTENT sizes
          // Copy should match the source size exactly
          dst_size = send_size_per_peer;  // SAME as source
          dst_index = target_peer;
          
          // Calculate destination offset in output buffer
          dst_offset = 0;
          for (int sender = 0; sender < target_peer; ++sender) {
            dst_offset += (sender + 1) * 1024; // accumulate sizes from previous senders
          }
          
        } else if (op_name == "put") {
          // For PUT: rank_ sends to target_peer - MUST BE SAME SIZE AS SOURCE
          dst_size = send_size_per_peer;  // SAME as source - CRITICAL!
          dst_index = target_peer;  // CRITICAL FIX: Use target_peer as index
          
          // PUT destination offset: where target_peer receives data from rank_
          dst_offset = 0;
          for (int sender = 0; sender < rank_; ++sender) {  // Use rank_ instead of gpu_id
            dst_offset += (sender + 1) * 1024;
          }
          
        } else {
          // For other operations: use consistent sizing
          dst_size = send_size_per_peer;  // SAME as source to avoid mismatches
          dst_index = target_peer;  // CRITICAL FIX: Use target_peer as index
          dst_offset = 0;
          for (int sender = 0; sender < target_peer; ++sender) {
            dst_offset += (sender + 1) * 1024;
          }
        }

        size_t dst_size_elements = dst_size / sizeof(uint32_t);
        size_t dst_offset_elements = dst_offset / sizeof(uint32_t);        

        // // COMPLETELY REPLACE the buffer object - don't process dynamic fields
        // nlohmann::json proper_buffer_obj = {
        //   {"index", static_cast<uint32_t>(0)},      // MSCCLPP uint32_t compatibility
        //   {"size", static_cast<uint32_t>(dst_size_elements)}, // Convert bytes to elements
        //   {"offset", static_cast<uint32_t>(dst_offset_elements)}, // Convert bytes to elements
        //   {"type", "o"}  // Output buffer for COPY operations
        // };

        nlohmann::json proper_buffer_obj;
        // For PUT operations, dst_buff should use scratch buffer type
        if (op_name == "put") {
            // PUT operations write to remote scratch buffers
            proper_buffer_obj = {
                {"index", static_cast<uint32_t>(1)},  // Scratch buffers use index 1, not 0!
                {"size", static_cast<uint32_t>(dst_size_elements)},
                {"offset", static_cast<uint32_t>(dst_offset_elements)},
                {"type", "s"}  // Scratch buffer for PUT operations
            };
        } else {
            // COPY operations write to output buffers
            proper_buffer_obj = {
                {"index", static_cast<uint32_t>(0)},  // Output buffer uses index 0
                {"size", static_cast<uint32_t>(dst_size_elements)},
                {"offset", static_cast<uint32_t>(dst_offset_elements)},
                {"type", "o"}  // Output buffer for COPY operations
            };
        }
          
        buff_obj = proper_buffer_obj;
        
        std::cout << "Rank " << rank_ << ": CREATED dst_buff[" << i << "] = " 
                  << "{ index:" << dst_index << ", size:" << dst_size 
                  << ", offset:" << dst_offset << ", type:\"o\" } for " << op_name << std::endl;
      }
    }
  }

  std::cout << "Rank " << rank_ << ": Completed processing operation " << op_index 
            << " (" << op_name << ") with CORRECT buffer indices" << std::endl;
}

void DynamicExecutionPlan::expandThreadblocks(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Starting threadblock expansion..." << std::endl;
  
  if (!json_obj.contains("gpus") || !json_obj["gpus"].is_array()) {
    std::cout << "Rank " << rank_ << ": No GPUs array found, skipping threadblock expansion" << std::endl;
    return;
  }
  
  auto& gpus_array = json_obj["gpus"];
  
  for (size_t gpu_idx = 0; gpu_idx < gpus_array.size(); ++gpu_idx) {
    auto& gpu_json = gpus_array[gpu_idx];
    
    if (!gpu_json.contains("threadblocks") || !gpu_json["threadblocks"].is_array()) {
      std::cout << "Rank " << rank_ << ": GPU " << gpu_idx << " has no threadblocks array" << std::endl;
      continue;
    }
    
    auto& original_threadblocks = gpu_json["threadblocks"];
    std::cout << "Rank " << rank_ << ": GPU " << gpu_idx << " has " << original_threadblocks.size() 
              << " template threadblocks" << std::endl;
    
    // Get dynamic parameters for calculating expansion
    int maxThreadBlocks = 32; // Default value
    int blockSize = 32768;    // Default value
    
    if (json_obj.contains("dynamic_parameters")) {
      auto& dynParams = json_obj["dynamic_parameters"];
      
      // ROBUST TYPE HANDLING for max_thread_blocks
      if (dynParams.contains("max_thread_blocks")) {
        try {
          if (dynParams["max_thread_blocks"].is_string()) {
            std::string str_val = dynParams["max_thread_blocks"].get<std::string>();
            maxThreadBlocks = std::stoi(str_val);
            std::cout << "Rank " << rank_ << ": Converted string max_thread_blocks '" << str_val 
                      << "' to int " << maxThreadBlocks << std::endl;
          } else if (dynParams["max_thread_blocks"].is_number_integer()) {
            maxThreadBlocks = dynParams["max_thread_blocks"].get<int>();
            std::cout << "Rank " << rank_ << ": Used int max_thread_blocks = " << maxThreadBlocks << std::endl;
          } else {
            std::cout << "Rank " << rank_ << ": WARNING: max_thread_blocks has unexpected type, using default " 
                      << maxThreadBlocks << std::endl;
          }
        } catch (const std::exception& e) {
          std::cout << "Rank " << rank_ << ": ERROR converting max_thread_blocks: " << e.what() 
                    << ", using default " << maxThreadBlocks << std::endl;
        }
      }
      
      // ROBUST TYPE HANDLING for block_size
      if (dynParams.contains("block_size")) {
        try {
          if (dynParams["block_size"].is_string()) {
            std::string str_val = dynParams["block_size"].get<std::string>();
            blockSize = std::stoi(str_val);
            std::cout << "Rank " << rank_ << ": Converted string block_size '" << str_val 
                      << "' to int " << blockSize << std::endl;
          } else if (dynParams["block_size"].is_number_integer()) {
            blockSize = dynParams["block_size"].get<int>();
            std::cout << "Rank " << rank_ << ": Used int block_size = " << blockSize << std::endl;
          } else {
            std::cout << "Rank " << rank_ << ": WARNING: block_size has unexpected type, using default " 
                      << blockSize << std::endl;
          }
        } catch (const std::exception& e) {
          std::cout << "Rank " << rank_ << ": ERROR converting block_size: " << e.what() 
                    << ", using default " << blockSize << std::endl;
        }
      }
    }
    
    // Create expanded threadblocks array
    nlohmann::json expanded_threadblocks = nlohmann::json::array();
    
    // Calculate threadblocks per peer group
    int threadblocksPerPeer = std::max(1, maxThreadBlocks / static_cast<int>(original_threadblocks.size()));
    
    std::cout << "Rank " << rank_ << ": Creating " << threadblocksPerPeer 
              << " threadblocks per peer (maxThreadBlocks=" << maxThreadBlocks 
              << ", template_count=" << original_threadblocks.size() << ")" << std::endl;
    
    int global_tb_id = 0;
    
    // For each template threadblock (representing a peer in AllToAllV)
    for (size_t template_idx = 0; template_idx < original_threadblocks.size(); ++template_idx) {
      auto& template_tb = original_threadblocks[template_idx];
      
      // ROBUST TYPE HANDLING for dynamic_tbgroup_id
      int dynamic_tbgroup_id = -1;
      if (template_tb.contains("dynamic_tbgroup_id")) {
        try {
          if (template_tb["dynamic_tbgroup_id"].is_string()) {
            std::string str_val = template_tb["dynamic_tbgroup_id"].get<std::string>();
            dynamic_tbgroup_id = std::stoi(str_val);
            std::cout << "Rank " << rank_ << ": Converted string dynamic_tbgroup_id '" << str_val 
                      << "' to int " << dynamic_tbgroup_id << std::endl;
          } else if (template_tb["dynamic_tbgroup_id"].is_number_integer()) {
            dynamic_tbgroup_id = template_tb["dynamic_tbgroup_id"].get<int>();
          } else {
            std::cout << "Rank " << rank_ << ": WARNING: dynamic_tbgroup_id has unexpected type, using -1" << std::endl;
          }
        } catch (const std::exception& e) {
          std::cout << "Rank " << rank_ << ": ERROR converting dynamic_tbgroup_id: " << e.what() 
                    << ", using -1" << std::endl;
        }
      }
      
      std::cout << "Rank " << rank_ << ": Expanding template threadblock " << template_idx 
                << " (dynamic_tbgroup_id=" << dynamic_tbgroup_id << ") into " 
                << threadblocksPerPeer << " concrete threadblocks" << std::endl;
      
      // Create multiple threadblocks for this peer
      for (int tb_instance = 0; tb_instance < threadblocksPerPeer; ++tb_instance) {
        // Create a copy of the template threadblock
        nlohmann::json concrete_tb = template_tb;
        
        // Set the concrete threadblock ID
        concrete_tb["id"] = global_tb_id;
        
        // CRITICAL: Remove ALL template-specific and internal fields
        if (concrete_tb.contains("dynamic_tbgroup_id")) {
          concrete_tb.erase("dynamic_tbgroup_id");
        }
        if (concrete_tb.contains("tb_count")) {
          concrete_tb.erase("tb_count");
        }
        if (concrete_tb.contains("tb_group_id")) {
          concrete_tb.erase("tb_group_id");
        }
        // Remove the internal fields that shouldn't be in final plan
        if (concrete_tb.contains("tbgroup_id")) {
          concrete_tb.erase("tbgroup_id");
        }
        if (concrete_tb.contains("tbgroup_instance")) {
          concrete_tb.erase("tbgroup_instance");
        }
        
        std::cout << "Rank " << rank_ << ": Created concrete threadblock with id=" << global_tb_id 
                  << " from template " << template_idx << " instance " << tb_instance << std::endl;
        
        expanded_threadblocks.push_back(concrete_tb);
        global_tb_id++;
      }
    }
    
    // Replace the original threadblocks with the expanded version
    gpu_json["threadblocks"] = expanded_threadblocks;
    
    std::cout << "Rank " << rank_ << ": GPU " << gpu_idx << " expanded from " 
              << original_threadblocks.size() << " template threadblocks to " 
              << expanded_threadblocks.size() << " concrete threadblocks" << std::endl;
  }
  
  std::cout << "Rank " << rank_ << ": Completed threadblock expansion" << std::endl;
}

void DynamicExecutionPlan::sanitizeJsonForSerialization(JsonType& json_obj) {
  std::cout << "Rank " << rank_ << ": Sanitizing JSON for serialization..." << std::endl;
  
  // Comprehensive sanitization to ensure all numbers are regular int type
  std::function<void(nlohmann::json&)> sanitize = [&](nlohmann::json& j) {
    if (j.is_object()) {
      for (auto it = j.begin(); it != j.end(); ++it) {
        sanitize(it.value());
      }
    } else if (j.is_array()) {
      for (auto it = j.begin(); it != j.end(); ++it) {
        sanitize(*it);
      }
    } else if (j.is_number()) {
      // Convert all numbers to regular int to avoid type conflicts
      if (j.is_number_integer()) {
        int int_val = j.get<int>();
        j = int_val;  // This ensures it's stored as regular int, not int64_t
      }
    }
  };
  
  nlohmann::json& raw_json = static_cast<nlohmann::json&>(json_obj);
  sanitize(raw_json);
  
  std::cout << "Rank " << rank_ << ": Completed JSON sanitization with consistent int types" << std::endl;
}
}  // namespace mscclpp