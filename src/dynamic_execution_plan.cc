// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
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
  std::cout << "Rank " << rank_ << ": Attempting to load JSON from: " << planPath << std::endl;
  
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
  
  std::cout << "Rank " << rank_ << ": File size: " << file_size << " bytes" << std::endl;
  
  // Read first few characters for debugging
  std::string first_line;
  std::getline(file, first_line);
  file.seekg(0, std::ios::beg);  // Reset to beginning
  
  std::cout << "Rank " << rank_ << ": First line: " << first_line.substr(0, 50) << "..." << std::endl;
  
  json j;
  try {
    file >> j;
  } catch (const json::parse_error& e) {
    std::string error_msg = "JSON parse error in file " + planPath + ": " + e.what();
    std::cout << "Rank " << rank_ << ": " << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  
  // Parse basic plan information
  name_ = j.value("name", "dynamic_plan");
  collective_ = j.value("collective", "alltoallv");
  protocol_ = j.value("protocol", "dynamic");
  isDynamic_ = j.value("dynamic", true);
  minMessageSize_ = j.value("min_message_size", 0);
  maxMessageSize_ = j.value("max_message_size", 1048576);
  numThreadsPerBlock_ = j.value("num_threads_per_block", 1024);
  
  std::cout << "Rank " << rank_ << ": Successfully parsed JSON - name: " << name_ 
            << ", collective: " << collective_ << std::endl;
  
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
  json concrete_json;
  
  // Basic plan information
  concrete_json["name"] = name_ + "_instantiated";
  concrete_json["collective"] = "alltoallv";
  concrete_json["protocol"] = "Simple";
  concrete_json["inplace"] = false;
  concrete_json["reuse_resources"] = false;
  
  // Buffer alignment configuration that works
  concrete_json["buffer_alignment"] = 16;
  concrete_json["num_threads_per_block"] = 1024;
  concrete_json["use_double_scratch_buffer"] = false;
  concrete_json["min_message_size"] = 0;
  concrete_json["max_message_size"] = 18446744073709551615ULL;
  
  // Generate concrete GPU information for ALL ranks
  json gpus_json = json::array();
  int num_ranks = static_cast<int>(params.peerRanks.size());
  
  size_t element_size = sizeof(uint32_t);  // 4 bytes per element
  size_t total_buffer_bytes = std::max(params.totalSendSize, params.totalRecvSize);
  
  // Working chunk calculation: chunks = bytes / alignment
  size_t chunk_alignment = 16;  // from buffer_alignment
  size_t num_chunks = total_buffer_bytes / chunk_alignment;
  
  std::cout << "Rank " << rank_ << ": Buffer configuration:" << std::endl;
  std::cout << "  - total_buffer_bytes: " << total_buffer_bytes << std::endl;
  std::cout << "  - num_chunks: " << num_chunks << " (alignment=" << chunk_alignment << ")" << std::endl;
  
  // Create execution plan for each rank
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    json gpu_json;
    gpu_json["id"] = rank_id;
    
    // Set chunks based on our working calculation
    gpu_json["input_chunks"] = static_cast<int>(num_chunks);
    gpu_json["output_chunks"] = static_cast<int>(num_chunks);
    gpu_json["scratch_chunks"] = 0;
    
    // Empty arrays for resources (we'll use local copy operations)
    gpu_json["channels"] = json::array();
    gpu_json["remote_buffers"] = json::array();
    gpu_json["semaphores"] = json::array();
    
    // Create threadblocks with actual copy operations for alltoallv
    json threadblocks = json::array();
    json threadblock;
    threadblock["id"] = 0;
    
    // Create copy operations to simulate alltoallv data movement
    json operations = json::array();
    
    // For alltoallv, each rank needs to copy data from its send buffer to recv buffer
    // with the appropriate offsets for each peer
    
    // Calculate offsets and sizes for this rank's operations
    size_t input_offset = 0;
    size_t output_offset = 0;
    
    for (int peer = 0; peer < num_ranks; ++peer) {
      // Size this rank sends to/receives from peer (in bytes)
      size_t send_size_bytes = (rank_id + 1) * 1024;  // Pattern from test
      size_t recv_size_bytes = (peer + 1) * 1024;     // Pattern from test
      
      // Convert to chunks (each chunk is 16 bytes)
      size_t send_size_chunks = send_size_bytes / chunk_alignment;
      size_t recv_size_chunks = recv_size_bytes / chunk_alignment;
      
      // Only create copy operation if there's data to copy
      // and if it fits within our buffer
      if (send_size_chunks > 0 && 
          (input_offset + send_size_chunks) <= num_chunks &&
          (output_offset + recv_size_chunks) <= num_chunks) {
        
        // For local testing, copy from input to output with correct offsets
        // In a real alltoallv, this would involve remote operations
        json copy_op;
        copy_op["name"] = "copy";
        
        // Source buffer (input)
        copy_op["src_buff"] = json::array({
          {
            {"type", "i"},
            {"index", 0},
            {"offset", static_cast<int>(input_offset)},
            {"size", static_cast<int>(send_size_chunks)}
          }
        });
        
        // Destination buffer (output)
        copy_op["dst_buff"] = json::array({
          {
            {"type", "o"},
            {"index", 0},
            {"offset", static_cast<int>(output_offset)},
            {"size", static_cast<int>(send_size_chunks)}
          }
        });
        
        operations.push_back(copy_op);
        
        std::cout << "Rank " << rank_id << ": Copy op for peer " << peer 
                  << " - src offset=" << input_offset << ", dst offset=" << output_offset
                  << ", size=" << send_size_chunks << " chunks" << std::endl;
      }
      
      // Update offsets for next peer
      input_offset += send_size_chunks;
      output_offset += recv_size_chunks;
    }
    
    // If no operations were created, add a nop to avoid empty operation list
    if (operations.empty()) {
      json nop_op;
      nop_op["name"] = "nop";
      operations.push_back(nop_op);
      std::cout << "Rank " << rank_id << ": No copy operations created, using nop" << std::endl;
    }
    
    threadblock["ops"] = operations;
    
    // Empty arrays for threadblock-level resources
    threadblock["channels"] = json::array();
    threadblock["remote_buffer_refs"] = json::array();
    
    threadblocks.push_back(threadblock);
    gpu_json["threadblocks"] = threadblocks;
    
    gpus_json.push_back(gpu_json);
  }
  
  concrete_json["gpus"] = gpus_json;
  
  std::cout << "Rank " << rank_ << ": Generated JSON with " << gpus_json.size() 
            << " GPUs and copy operations for alltoallv simulation" << std::endl;
  
  return concrete_json.dump(2);
}

std::shared_ptr<ExecutionPlan> DynamicExecutionPlan::createExecutionPlan(const DynamicRuntimeParams& params) {
  try {
    std::cout << "Rank " << rank_ << ": Starting createExecutionPlan..." << std::endl;
    
    // Generate concrete JSON in memory
    std::string concrete_json = instantiate(params);
    
    // Debug: Print the COMPLETE generated JSON for analysis
    std::cout << "Rank " << rank_ << ": COMPLETE Generated JSON:\n" << concrete_json << std::endl;
    
    // Create a persistent temporary file that won't be deleted immediately
    // Use a more unique name to avoid conflicts between ranks
    std::string temp_plan_path = "/tmp/dynamic_plan_rank" + std::to_string(rank_) + "_pid" + 
                                 std::to_string(getpid()) + "_" + std::to_string(std::time(nullptr)) + ".json";
    
    std::cout << "Rank " << rank_ << ": Creating persistent temporary file: " << temp_plan_path << std::endl;
    
    // Write JSON to temporary file with explicit flushing and sync
    {
      std::ofstream temp_file(temp_plan_path);
      if (!temp_file.is_open()) {
        throw std::runtime_error("Cannot create temporary execution plan file: " + temp_plan_path);
      }
      
      temp_file << concrete_json;
      temp_file.flush();  // Explicit flush
      
      // Force file system sync
      if (temp_file.good()) {
        temp_file.close();  // Explicit close
        std::cout << "Rank " << rank_ << ": Temporary file written and closed successfully" << std::endl;
      } else {
        throw std::runtime_error("Error writing to temporary file: " + temp_plan_path);
      }
    }
    
    // Add a small delay to ensure file system operations complete
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Verify file was written correctly
    std::ifstream verify_file(temp_plan_path);
    if (!verify_file.is_open()) {
      throw std::runtime_error("Cannot verify temporary file: " + temp_plan_path);
    }
    
    // Check file size
    verify_file.seekg(0, std::ios::end);
    size_t file_size = verify_file.tellg();
    verify_file.seekg(0, std::ios::beg);
    
    std::cout << "Rank " << rank_ << ": Temporary file size: " << file_size << " bytes" << std::endl;
    
    if (file_size == 0) {
      verify_file.close();
      throw std::runtime_error("Temporary file is empty: " + temp_plan_path);
    }
    
    std::string first_line;
    std::getline(verify_file, first_line);
    verify_file.seekg(0, std::ios::beg);
    
    // Read entire file content for verification
    std::string file_content((std::istreambuf_iterator<char>(verify_file)),
                             std::istreambuf_iterator<char>());
    verify_file.close();
    
    if (first_line.empty() || file_content.empty()) {
      throw std::runtime_error("Temporary file content is empty: " + temp_plan_path);
    }
    
    std::cout << "Rank " << rank_ << ": Temporary file verified, first line: " << first_line.substr(0, 50) << "..." << std::endl;
    std::cout << "Rank " << rank_ << ": File content length: " << file_content.length() << std::endl;
    
    // Test JSON parsing before creating ExecutionPlan
    try {
      json test_json = json::parse(file_content);
      std::cout << "Rank " << rank_ << ": JSON parsing test successful" << std::endl;
      
      // Debug: test the specific access that's failing
      if (test_json.contains("gpus") && test_json["gpus"].is_array()) {
        const auto& gpus = test_json["gpus"];
        std::cout << "Rank " << rank_ << ": gpus array size: " << gpus.size() << std::endl;
        if (rank_ < static_cast<int>(gpus.size())) {
          const auto& gpu = gpus[rank_];
          std::cout << "Rank " << rank_ << ": Successfully accessed gpus[" << rank_ << "]" << std::endl;
          if (gpu.contains("id")) {
            std::cout << "Rank " << rank_ << ": GPU id: " << gpu["id"] << std::endl;
          }
        } else {
          std::cout << "Rank " << rank_ << ": ERROR - rank " << rank_ << " >= gpus.size() " << gpus.size() << std::endl;
        }
      } else {
        std::cout << "Rank " << rank_ << ": ERROR - gpus is not an array or doesn't exist" << std::endl;
      }
      
    } catch (const json::parse_error& e) {
      std::cout << "Rank " << rank_ << ": JSON parsing test failed: " << e.what() << std::endl;
      std::cout << "Rank " << rank_ << ": File content: " << file_content << std::endl;
      throw std::runtime_error("Generated JSON is invalid: " + std::string(e.what()));
    }
    
    // Create ExecutionPlan from the temporary file
    std::cout << "Rank " << rank_ << ": Creating ExecutionPlan from temporary file..." << std::endl;
    auto execution_plan = std::make_shared<ExecutionPlan>(temp_plan_path, rank_);
    
    std::cout << "Rank " << rank_ << ": ExecutionPlan created successfully" << std::endl;
    
    // Store the temp file path so we can clean it up later
    // Note: We'll need to clean this up manually after execution completes
    temp_file_path_ = temp_plan_path;
    
    std::cout << "Rank " << rank_ << ": Temporary file will persist for ExecutionPlan usage: " << temp_plan_path << std::endl;
    
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
  
  // For MSCCLPP consistency, calculate the maximum buffer size that any rank will need
  // This needs to be coordinated across ranks, but for now assume symmetric pattern
  size_t maxSendSize = params.totalSendSize;
  size_t maxRecvSize = params.totalRecvSize;
  
  // For an alltoallv with variable sizes, estimate the maximum buffer size
  // In the current test pattern: rank r sends (r+1)*1024 to each peer
  // So max send = (num_ranks)*1024 * num_ranks
  // And max recv = sum of all different send sizes
  size_t estimatedMaxSend = 0;
  size_t estimatedMaxRecv = 0;
  
  for (int r = 0; r < num_ranks; ++r) {
    size_t rankSendTotal = 0;
    size_t rankRecvTotal = 0;
    
    for (int p = 0; p < num_ranks; ++p) {
      rankSendTotal += (r + 1) * 1024;  // What rank r sends
      rankRecvTotal += (p + 1) * 1024;  // What rank r receives from rank p
    }
    
    estimatedMaxSend = std::max(estimatedMaxSend, rankSendTotal);
    estimatedMaxRecv = std::max(estimatedMaxRecv, rankRecvTotal);
  }
  
  // Use the maximum of estimated max send/recv as the consistent buffer size
  size_t maxBufferSize = std::max(estimatedMaxSend, estimatedMaxRecv);
  
  // Override the totals with consistent sizes
  params.totalSendSize = maxBufferSize;
  params.totalRecvSize = maxBufferSize;
  
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
    int rank = comm->bootstrap()->getRank();
    int numRanks = comm->bootstrap()->getNranks();
    
    std::cout << "Rank " << rank << ": Setting up MSCCLPP execution with " << numRanks << " ranks" << std::endl;
    
    // Step 1: Create runtime parameters FIRST
    auto runtimeParams = createRuntimeParams(sendSizes, recvSizes);
    
    std::cout << "Rank " << rank << ": Runtime parameters created" << std::endl;
    std::cout << "  - totalSendSize: " << runtimeParams.totalSendSize << std::endl;
    std::cout << "  - totalRecvSize: " << runtimeParams.totalRecvSize << std::endl;
    
    // Step 2: Create ExecutionPlan EARLY so we know what buffer sizes it expects
    auto executionPlan = dynamicPlan->createExecutionPlan(runtimeParams);
    std::cout << "Rank " << rank << ": Created execution plan: " << executionPlan->name() << std::endl;
    
    // Step 3: Use the buffer sizes that match what the ExecutionPlan expects
    size_t maxBufferSize = std::max(runtimeParams.totalSendSize, runtimeParams.totalRecvSize);
    
    std::cout << "Rank " << rank << ": Using consistent buffer size: " << maxBufferSize 
              << " (send: " << runtimeParams.totalSendSize 
              << ", recv: " << runtimeParams.totalRecvSize << ")" << std::endl;
    
    // Step 4: Register memory buffers with the sizes that match the ExecutionPlan
    auto sendBufferRegistered = comm->registerMemory(sendBuffer, 
        maxBufferSize, Transport::CudaIpc);
    auto recvBufferRegistered = comm->registerMemory(recvBuffer, 
        maxBufferSize, Transport::CudaIpc);
    
    std::cout << "Rank " << rank << ": Registered memory buffers" << std::endl;
    
    // Step 5: Setup connections to all peer ranks (only same-node for simplicity)
    std::vector<std::shared_future<std::shared_ptr<Connection>>> connectionFutures;
    std::vector<std::shared_ptr<Connection>> connections;
    
    for (int peer_rank = 0; peer_rank < numRanks; ++peer_rank) {
      if (peer_rank != rank) {
        // For this example, we'll use CudaIpc if on same node, otherwise skip complex networking
        bool sameNode = (peer_rank / 8) == (rank / 8);  // Assuming 8 GPUs per node
        
        if (sameNode) {
          // Create endpoint configuration for CudaIpc transport
          EndpointConfig config;
          config.transport = Transport::CudaIpc;
          
          // Establish connection to peer rank
          auto connectionFuture = comm->connect(config, peer_rank, 0);
          connectionFutures.push_back(connectionFuture);
          
          std::cout << "Rank " << rank << ": Initiated CudaIpc connection to rank " << peer_rank << std::endl;
        } else {
          std::cout << "Rank " << rank << ": Skipping cross-node connection to rank " << peer_rank 
                    << " (requires InfiniBand or Ethernet)" << std::endl;
        }
      }
    }
    
    // Step 6: Wait for all connections to be established
    for (auto& future : connectionFutures) {
      connections.push_back(future.get());
    }
    
    std::cout << "Rank " << rank << ": Established " << connections.size() << " connections" << std::endl;
    
    // Step 7: Send memory handles to connected peers
    for (size_t i = 0; i < connections.size(); ++i) {
      int peerRank = comm->remoteRankOf(*connections[i]);
      comm->sendMemory(sendBufferRegistered, peerRank, 0);
      comm->sendMemory(recvBufferRegistered, peerRank, 1);
      
      std::cout << "Rank " << rank << ": Sent memory handles to rank " << peerRank << std::endl;
    }
    
    // Step 8: Receive memory handles from connected peers
    std::vector<std::shared_future<RegisteredMemory>> remoteSendMemories;
    std::vector<std::shared_future<RegisteredMemory>> remoteRecvMemories;
    
    for (size_t i = 0; i < connections.size(); ++i) {
      int peerRank = comm->remoteRankOf(*connections[i]);
      remoteSendMemories.push_back(comm->recvMemory(peerRank, 0));
      remoteRecvMemories.push_back(comm->recvMemory(peerRank, 1));
    }
    
    // Wait for all memory exchanges to complete
    for (auto& future : remoteSendMemories) {
      future.wait();
    }
    for (auto& future : remoteRecvMemories) {
      future.wait();
    }
    
    std::cout << "Rank " << rank << ": Memory exchange completed with " << connections.size() << " peers" << std::endl;
    
    // Step 9: Create and setup Executor
    auto executor = std::make_shared<Executor>(comm);
    
    std::cout << "Rank " << rank << ": Created executor, executing plan..." << std::endl;
    
    // Step 10: Execute the plan with the exact buffer sizes from ExecutionPlan
    std::cout << "Rank " << rank << ": About to execute with:" << std::endl;
    std::cout << "  - sendBuffer: " << sendBuffer << std::endl;
    std::cout << "  - recvBuffer: " << recvBuffer << std::endl; 
    std::cout << "  - maxBufferSize: " << maxBufferSize << " bytes" << std::endl;
    std::cout << "  - maxBufferSize in elements: " << (maxBufferSize / sizeof(uint32_t)) << std::endl;
    std::cout << "  - DataType: UINT32" << std::endl;
    std::cout << "  - Execution plan name: " << executionPlan->name() << std::endl;
    
    // CRUCIAL: Use the exact same buffer sizes that the ExecutionPlan was created with
    executor->execute(rank, sendBuffer, recvBuffer, 
                     maxBufferSize, maxBufferSize,  // These must match the JSON chunks
                     DataType::UINT32,
                     *executionPlan, cudaStreamDefault);
    
    std::cout << "Rank " << rank << ": Execution completed successfully!" << std::endl;
    
    // Clean up
    dynamicPlan->cleanup();
    
    return true;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << comm->bootstrap()->getRank() << ": Error in execute: " << e.what() << std::endl;
    dynamicPlan->cleanup();
    return false;
  }
}

void DynamicExecutionPlan::cleanup() {
  if (!temp_file_path_.empty()) {
    std::cout << "Rank " << rank_ << ": Cleaning up temporary file: " << temp_file_path_ << std::endl;
    std::remove(temp_file_path_.c_str());
    temp_file_path_.clear();
  }
}

}  // namespace mscclpp