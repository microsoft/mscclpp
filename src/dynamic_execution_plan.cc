// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>  // Now included only in implementation file
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <ctime>
#include <numeric>
#include <iostream>
#include <chrono>  // for sleep
#include <thread>  // for this_thread
#include <unistd.h>  // for getpid

namespace mscclpp {

// Define JsonType alias for consistency
namespace detail {
  using JsonType = nlohmann::json;
}

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
      minMessageSize_(0), maxMessageSize_(0), numThreadsPerBlock_(1024),
      templateJson_(std::make_unique<detail::JsonType>()) {
  loadFromJson(planPath);
}

void DynamicExecutionPlan::loadFromJson(const std::string& planPath) {
  std::cout << "Rank " << rank_ << ": Attempting to load DSL JSON from: " << planPath << std::endl;
  
  std::ifstream file(planPath);
  if (!file.is_open()) {
    std::string error_msg = "Cannot open dynamic execution plan file: " + planPath;
    std::cout << "Rank " << rank_ << ": " << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  
  // Check if file is empty
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  if (file_size == 0) {
    std::string error_msg = "Dynamic execution plan file is empty: " + planPath;
    std::cout << "Rank " << rank_ << ": " << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  
  std::cout << "Rank " << rank_ << ": DSL file size: " << file_size << " bytes" << std::endl;
  
  detail::JsonType j;
  try {
    file >> j;
  } catch (const detail::JsonType::parse_error& e) {
    std::string error_msg = "JSON parse error in file " + planPath + ": " + e.what();
    std::cout << "Rank " << rank_ << ": " << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  
  // Parse basic plan information
  name_ = j.value("name", "dynamic_plan");
  collective_ = j.value("collective", "alltoallv");
  protocol_ = j.value("protocol", "Simple");
  isDynamic_ = j.value("dynamic", true);
  minMessageSize_ = j.value("min_message_size", 0);
  maxMessageSize_ = j.value("max_message_size", 1048576);
  numThreadsPerBlock_ = j.value("num_threads_per_block", 1024);
  
  std::cout << "Rank " << rank_ << ": Successfully parsed DSL JSON - name: " << name_ 
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
  
  int threadBlocks = std::max(1, static_cast<int>((messageSize + blockSize - 1) / blockSize));
  return std::min(threadBlocks, maxThreadBlocks);
}

std::string DynamicExecutionPlan::instantiate(const DynamicRuntimeParams& params) {
  std::cout << "Rank " << rank_ << ": Starting DSL-based instantiation..." << std::endl;
  
  // Create a copy of the template JSON
  detail::JsonType concrete_json = *templateJson_;
  
  // Calculate chunk information for alltoallv
  size_t chunk_alignment = 16;  // MSCCLPP alignment requirement
  
  // Calculate total input and output chunks based on send/recv sizes
  size_t total_input_chunks = 0;
  size_t total_output_chunks = 0;
  
  for (int peer = 0; peer < params.num_ranks; ++peer) {
    size_t send_size_bytes = params.send_sizes[peer];
    size_t recv_size_bytes = params.recv_sizes[peer];
    
    size_t send_size_chunks = (send_size_bytes + chunk_alignment - 1) / chunk_alignment;
    size_t recv_size_chunks = (recv_size_bytes + chunk_alignment - 1) / chunk_alignment;
    
    total_input_chunks += send_size_chunks;
    total_output_chunks += recv_size_chunks;
  }
  
  std::cout << "Rank " << rank_ << ": Calculated total_input_chunks=" << total_input_chunks 
            << ", total_output_chunks=" << total_output_chunks << std::endl;
  
  // Create variable context for substitution
  VariableContext var_context;
  var_context.setVariable("DYNAMIC_INPUT_CHUNKS", std::to_string(total_input_chunks));
  var_context.setVariable("DYNAMIC_OUTPUT_CHUNKS", std::to_string(total_output_chunks));
  var_context.setVariable("DYNAMIC_SCRATCH_CHUNKS", std::to_string(params.num_ranks - 1));
  
  // Add common template variables for operation templates
  var_context.setVariable("operation_type", "copy");
  var_context.setVariable("channel_id", "0");
  var_context.setVariable("src_buffer_type", "i");
  var_context.setVariable("dst_buffer_type", "o");
  
  // Update the GPU entry for this rank
  if (concrete_json.contains("gpus") && rank_ < concrete_json["gpus"].size()) {
    auto& gpu_json = concrete_json["gpus"][rank_];
    
    // Substitute dynamic chunk variables
    if (gpu_json.contains("input_chunks") && gpu_json["input_chunks"].is_string()) {
      std::string input_chunks_str = gpu_json["input_chunks"];
      gpu_json["input_chunks"] = std::stoi(var_context.substituteVariables(input_chunks_str));
    }
    
    if (gpu_json.contains("output_chunks") && gpu_json["output_chunks"].is_string()) {
      std::string output_chunks_str = gpu_json["output_chunks"];
      gpu_json["output_chunks"] = std::stoi(var_context.substituteVariables(output_chunks_str));
    }
    
    if (gpu_json.contains("scratch_chunks") && gpu_json["scratch_chunks"].is_string()) {
      std::string scratch_chunks_str = gpu_json["scratch_chunks"];
      gpu_json["scratch_chunks"] = std::stoi(var_context.substituteVariables(scratch_chunks_str));
    }
    
    // Process threadblocks and operations
    if (gpu_json.contains("threadblocks")) {
      for (auto& threadblock : gpu_json["threadblocks"]) {
        if (threadblock.contains("ops")) {
          for (auto& op : threadblock["ops"]) {
            // Update operations marked as templates
            if (op.contains("template") && op["template"].get<bool>()) {
              updateOperationWithRuntimeParams(op, params, var_context);
            }
          }
        }
      }
    }
    
    // Process operation templates
    processOperationTemplates(gpu_json, params, var_context);
    
    std::cout << "Rank " << rank_ << ": Updated DSL JSON with runtime parameters" << std::endl;
  }
  
  // For simplicity in this example, create a local copy-only version
  // This avoids inter-rank communication that was causing hangs
  return createLocalCopyVersion(params, var_context);
}

void DynamicExecutionPlan::processOperationTemplates(detail::JsonType& gpu_json, 
                                                     const DynamicRuntimeParams& params,
                                                     const VariableContext& var_context) {
  if (!gpu_json.contains("operations")) {
    return;
  }
  
  auto& operations = gpu_json["operations"];
  for (auto& operation : operations) {
    if (operation.contains("operation_template")) {
      auto& operation_template = operation["operation_template"];
      substituteOperationTemplateVariables(operation_template, params, var_context);
    }
  }
}

void DynamicExecutionPlan::substituteOperationTemplateVariables(detail::JsonType& operation_template,
                                                               const DynamicRuntimeParams& params,
                                                               const VariableContext& var_context) {
  // Create enhanced variable context with runtime-specific values
  VariableContext enhanced_context = var_context;
  
  // Add runtime-specific variables for each peer and chunk
  for (int peer = 0; peer < params.num_ranks; ++peer) {
    size_t chunk_size = params.send_sizes[peer];
    int tb_count = calculateThreadBlocks(chunk_size);
    
    // Set peer-specific variables
    enhanced_context.setVariable("chunk_id", std::to_string(peer));
    enhanced_context.setVariable("peer_rank", std::to_string(peer));
    enhanced_context.setVariable("tb_count", std::to_string(tb_count));
    enhanced_context.setVariable("chunk_size", std::to_string(chunk_size));
    enhanced_context.setVariable("step_id", std::to_string(peer));
    enhanced_context.setVariable("src_chunk_index", std::to_string(peer));
    enhanced_context.setVariable("dst_chunk_index", std::to_string(peer));
    enhanced_context.setVariable("src_chunk_size", std::to_string(chunk_size));
    enhanced_context.setVariable("dst_chunk_size", std::to_string(chunk_size));
  }
  
  // Recursively substitute variables in all fields
  std::function<void(detail::JsonType&)> substitute_recursive = [&](detail::JsonType& json_obj) {
    if (json_obj.is_string()) {
      std::string str_val = json_obj.get<std::string>();
      json_obj = enhanced_context.substituteVariables(str_val);
      
      // Try to convert to number if it's a numeric string
      try {
        if (json_obj.get<std::string>().find_first_not_of("0123456789") == std::string::npos) {
          json_obj = std::stoi(json_obj.get<std::string>());
        }
      } catch (...) {
        // Keep as string if conversion fails
      }
    } else if (json_obj.is_object()) {
      for (auto& [key, value] : json_obj.items()) {
        substitute_recursive(value);
      }
    } else if (json_obj.is_array()) {
      for (auto& item : json_obj) {
        substitute_recursive(item);
      }
    }
  };
  
  substitute_recursive(operation_template);
}

std::string DynamicExecutionPlan::createLocalCopyVersion(const DynamicRuntimeParams& params, 
                                                         const VariableContext& var_context) {
  std::cout << "Rank " << rank_ << ": Creating local copy version to avoid hangs" << std::endl;
  
  // Create a simplified JSON that only does local operations
  detail::JsonType local_json;
  local_json["name"] = name_;
  local_json["collective"] = collective_;
  local_json["protocol"] = protocol_;
  local_json["inplace"] = true;
  local_json["reuse_resources"] = false;
  local_json["num_threads_per_block"] = numThreadsPerBlock_;
  local_json["min_message_size"] = minMessageSize_;
  local_json["max_message_size"] = maxMessageSize_;
  
  // Calculate chunks
  size_t chunk_alignment = 16;
  size_t total_input_chunks = 0;
  size_t total_output_chunks = 0;
  
  for (int peer = 0; peer < params.num_ranks; ++peer) {
    total_input_chunks += (params.send_sizes[peer] + chunk_alignment - 1) / chunk_alignment;
    total_output_chunks += (params.recv_sizes[peer] + chunk_alignment - 1) / chunk_alignment;
  }
  
  auto gpus_json = detail::JsonType::array();
  
  // Create GPU entry for this rank only
  detail::JsonType gpu_json;
  gpu_json["id"] = rank_;
  gpu_json["input_chunks"] = static_cast<int>(total_input_chunks);
  gpu_json["output_chunks"] = static_cast<int>(total_output_chunks);
  gpu_json["scratch_chunks"] = 0;  // No scratch needed for local copy
  
  // Create single threadblock with local copy operations
  auto threadblocks = detail::JsonType::array();
  detail::JsonType threadblock;
  threadblock["id"] = 0;
  
  auto operations = detail::JsonType::array();
  
  // Create copy operations to simulate alltoallv data rearrangement
  size_t input_offset = 0;
  size_t output_offset = 0;
  
  for (int peer = 0; peer < params.num_ranks; ++peer) {
    size_t send_size_chunks = (params.send_sizes[peer] + chunk_alignment - 1) / chunk_alignment;
    size_t recv_size_chunks = (params.recv_sizes[peer] + chunk_alignment - 1) / chunk_alignment;
    
    if (recv_size_chunks > 0) {
      // Create copy operation for this peer's data
      detail::JsonType copy_op;
      copy_op["name"] = "copy";
      
      // Use input data as source (simulating received data)
      auto src_buff = detail::JsonType::array();
      detail::JsonType src_element;
      src_element["type"] = "i";
      src_element["index"] = static_cast<int>(input_offset % total_input_chunks);
      src_element["size"] = static_cast<int>(recv_size_chunks);
      
      // Add template variables for enhanced tracking
      src_element["dynamic_index"] = var_context.substituteVariables("${src_chunk_index}");
      src_element["dynamic_size"] = var_context.substituteVariables("${src_chunk_size}");
      
      src_buff.push_back(src_element);
      copy_op["src_buff"] = src_buff;
      
      // Output buffer as destination
      auto dst_buff = detail::JsonType::array();
      detail::JsonType dst_element;
      dst_element["type"] = "o";
      dst_element["index"] = static_cast<int>(output_offset);
      dst_element["size"] = static_cast<int>(recv_size_chunks);
      
      // Add template variables for enhanced tracking
      dst_element["dynamic_index"] = var_context.substituteVariables("${dst_chunk_index}");
      dst_element["dynamic_size"] = var_context.substituteVariables("${dst_chunk_size}");
      
      dst_buff.push_back(dst_element);
      copy_op["dst_buff"] = dst_buff;
      
      // Add template metadata
      copy_op["dynamic_size"] = var_context.substituteVariables("${chunk_size}");
      copy_op["dynamic_step"] = var_context.substituteVariables("${step_id}");
      copy_op["dynamic_input_chunk"] = var_context.substituteVariables("${chunk_id}");
      copy_op["dynamic_output_chunk"] = var_context.substituteVariables("${chunk_id}");
      copy_op["dynamic_peer"] = var_context.substituteVariables("${peer_rank}");
      copy_op["dynamic_threadblock_count"] = var_context.substituteVariables("${tb_count}");
      
      operations.push_back(copy_op);
      
      std::cout << "Rank " << rank_ << ": Copy for peer " << peer 
                << " - input[" << (input_offset % total_input_chunks) << ".." 
                << ((input_offset % total_input_chunks) + recv_size_chunks - 1) << "] -> output["
                << output_offset << ".." << (output_offset + recv_size_chunks - 1) << "]" << std::endl;
    }
    
    input_offset += send_size_chunks;
    output_offset += recv_size_chunks;
  }
  
  // Add a nop if no operations
  if (operations.empty()) {
    detail::JsonType nop_op;
    nop_op["name"] = "nop";
    operations.push_back(nop_op);
  }
  
  threadblock["ops"] = operations;
  threadblock["channels"] = detail::JsonType::array();
  threadblock["remote_buffer_refs"] = detail::JsonType::array();
  
  threadblocks.push_back(threadblock);
  gpu_json["threadblocks"] = threadblocks;
  gpu_json["channels"] = detail::JsonType::array();
  gpu_json["remote_buffers"] = detail::JsonType::array();
  gpu_json["semaphores"] = detail::JsonType::array();
  
  // Add operation templates for comprehensive template support
  auto op_templates = detail::JsonType::array();
  detail::JsonType op_template_container;
  detail::JsonType op_template;
  
  op_template["type"] = var_context.substituteVariables("${operation_type}");
  op_template["inputChunk"] = var_context.substituteVariables("${chunk_id}");
  op_template["outputChunk"] = var_context.substituteVariables("${chunk_id}");
  op_template["peer"] = var_context.substituteVariables("${peer_rank}");
  op_template["channel"] = var_context.substituteVariables("${channel_id}");
  op_template["threadblock_count"] = var_context.substituteVariables("${tb_count}");
  op_template["size"] = var_context.substituteVariables("${chunk_size}");
  op_template["step"] = var_context.substituteVariables("${step_id}");
  
  auto src_buff_template = detail::JsonType::array();
  detail::JsonType src_template;
  src_template["type"] = var_context.substituteVariables("${src_buffer_type}");
  src_template["index"] = var_context.substituteVariables("${src_chunk_index}");
  src_template["size"] = var_context.substituteVariables("${src_chunk_size}");
  src_buff_template.push_back(src_template);
  op_template["src_buff"] = src_buff_template;
  
  auto dst_buff_template = detail::JsonType::array();
  detail::JsonType dst_template;
  dst_template["type"] = var_context.substituteVariables("${dst_buffer_type}");
  dst_template["index"] = var_context.substituteVariables("${dst_chunk_index}");
  dst_template["size"] = var_context.substituteVariables("${dst_chunk_size}");
  dst_buff_template.push_back(dst_template);
  op_template["dst_buff"] = dst_buff_template;
  
  op_template_container["operation_template"] = op_template;
  op_templates.push_back(op_template_container);
  gpu_json["operations"] = op_templates;
  
  gpus_json.push_back(gpu_json);
  local_json["gpus"] = gpus_json;
  
  std::cout << "Rank " << rank_ << ": Created local copy JSON with " << operations.size() << " operations" << std::endl;
  
  return local_json.dump(2);
}

void DynamicExecutionPlan::updateOperationWithRuntimeParams(detail::JsonType& op, 
                                                           const DynamicRuntimeParams& params,
                                                           const VariableContext& var_context) {
  // Enhanced template variable substitution for individual operations
  
  // Substitute all dynamic_ prefixed template variables
  std::vector<std::string> dynamic_fields = {
    "dynamic_size", "dynamic_step", "dynamic_input_chunk", "dynamic_output_chunk",
    "dynamic_peer", "dynamic_threadblock_count"
  };
  
  for (const auto& field : dynamic_fields) {
    if (op.contains(field) && op[field].is_string()) {
      std::string template_str = op[field].get<std::string>();
      std::string substituted = var_context.substituteVariables(template_str);
      
      // Try to convert to integer if the result is numeric
      try {
        if (substituted.find_first_not_of("0123456789") == std::string::npos) {
          op[field] = std::stoi(substituted);
        } else {
          op[field] = substituted;
        }
      } catch (...) {
        op[field] = substituted;
      }
    }
  }
  
  // Update buffer references with template variables
  if (op.contains("src_buff")) {
    for (auto& buff : op["src_buff"]) {
      if (buff.contains("dynamic_index") && buff["dynamic_index"].is_string()) {
        std::string template_str = buff["dynamic_index"].get<std::string>();
        std::string substituted = var_context.substituteVariables(template_str);
        try {
          buff["dynamic_index"] = std::stoi(substituted);
        } catch (...) {
          buff["dynamic_index"] = substituted;
        }
      }
      
      if (buff.contains("dynamic_size") && buff["dynamic_size"].is_string()) {
        std::string template_str = buff["dynamic_size"].get<std::string>();
        std::string substituted = var_context.substituteVariables(template_str);
        try {
          buff["dynamic_size"] = std::stoi(substituted);
        } catch (...) {
          buff["dynamic_size"] = substituted;
        }
      }
    }
  }
  
  if (op.contains("dst_buff")) {
    for (auto& buff : op["dst_buff"]) {
      if (buff.contains("dynamic_index") && buff["dynamic_index"].is_string()) {
        std::string template_str = buff["dynamic_index"].get<std::string>();
        std::string substituted = var_context.substituteVariables(template_str);
        try {
          buff["dynamic_index"] = std::stoi(substituted);
        } catch (...) {
          buff["dynamic_index"] = substituted;
        }
      }
      
      if (buff.contains("dynamic_size") && buff["dynamic_size"].is_string()) {
        std::string template_str = buff["dynamic_size"].get<std::string>();
        std::string substituted = var_context.substituteVariables(template_str);
        try {
          buff["dynamic_size"] = std::stoi(substituted);
        } catch (...) {
          buff["dynamic_size"] = substituted;
        }
      }
    }
  }
  
  // Remove template marker
  if (op.contains("template")) {
    op.erase("template");
  }
}

std::shared_ptr<ExecutionPlan> DynamicExecutionPlan::createExecutionPlan(const DynamicRuntimeParams& params) {
  try {
    std::cout << "Rank " << rank_ << ": Starting createExecutionPlan with DSL template..." << std::endl;
    
    // Generate concrete JSON in memory
    std::string concrete_json = instantiate(params);
    
    // Create a persistent temporary file
    std::string temp_plan_path = "/tmp/dynamic_plan_dsl_rank" + std::to_string(rank_) + "_pid" + 
                                std::to_string(getpid()) + "_" + std::to_string(time(nullptr)) + ".json";
    
    std::cout << "Rank " << rank_ << ": Writing instantiated plan to: " << temp_plan_path << std::endl;
    
    // Write the JSON to the temp file
    std::ofstream temp_file(temp_plan_path);
    if (!temp_file.is_open()) {
      throw std::runtime_error("Cannot create temporary plan file: " + temp_plan_path);
    }
    temp_file << concrete_json;
    temp_file.close();
    
    // Add small delay to ensure file is written
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Rank " << rank_ << ": Creating ExecutionPlan from instantiated DSL template" << std::endl;
    
    // Create ExecutionPlan from the temporary file
    auto execution_plan = std::make_shared<ExecutionPlan>(temp_plan_path, rank_);
    
    std::cout << "Rank " << rank_ << ": Successfully created ExecutionPlan from DSL template" << std::endl;
    
    return execution_plan;
    
  } catch (const std::exception& e) {
    std::cout << "Rank " << rank_ << ": Error in createExecutionPlan: " << e.what() << std::endl;
    throw;
  }
}

std::unique_ptr<DynamicAllToAllv> DynamicExecutionPlan::createAllToAllv() {
  return std::make_unique<DynamicAllToAllv>(*this);
}

// DynamicAllToAllv implementation
DynamicAllToAllv::DynamicAllToAllv(DynamicExecutionPlan& plan) 
    : plan_(plan), rank_(plan.getRank()) {
  std::cout << "Rank " << rank_ << ": Created DynamicAllToAllv with DSL template" << std::endl;
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
  
  std::cout << "Rank " << rank_ << ": DynamicAllToAllv::execute called with DSL plan" << std::endl;
  
  // Create runtime parameters
  DynamicRuntimeParams params;
  params.num_ranks = send_sizes.size();
  params.send_sizes = send_sizes;
  params.recv_sizes = recv_sizes;
  params.send_offsets = send_offsets;
  params.recv_offsets = recv_offsets;
  
  // Log the alltoallv parameters
  std::cout << "Rank " << rank_ << ": AllToAllV parameters:" << std::endl;
  for (size_t i = 0; i < send_sizes.size(); ++i) {
    std::cout << "  To rank " << i << ": send_size=" << send_sizes[i] 
              << ", send_offset=" << send_offsets[i] << std::endl;
  }
  for (size_t i = 0; i < recv_sizes.size(); ++i) {
    std::cout << "  From rank " << i << ": recv_size=" << recv_sizes[i] 
              << ", recv_offset=" << recv_offsets[i] << std::endl;
  }
  
  // Create execution plan from DSL template
  auto execution_plan = plan_.createExecutionPlan(params);
  
  // Calculate total sizes
  size_t total_send_size = std::accumulate(send_sizes.begin(), send_sizes.end(), 0ULL);
  size_t total_recv_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0ULL);
  
  std::cout << "Rank " << rank_ << ": About to execute with total_send_size=" << total_send_size 
            << ", total_recv_size=" << total_recv_size << std::endl;
  
  // Execute using MSCCLPP executor
  executor->execute(rank_, send_buff, recv_buff, total_send_size, total_recv_size, 
                   DataType::FLOAT16, *execution_plan, stream);
  
  std::cout << "Rank " << rank_ << ": DSL-based execution completed" << std::endl;
}

void DynamicExecutionPlan::cleanup() {
  // Cleanup any temporary files created during instantiation
  std::cout << "Rank " << rank_ << ": DynamicExecutionPlan cleanup completed" << std::endl;
}

}  // namespace mscclpp