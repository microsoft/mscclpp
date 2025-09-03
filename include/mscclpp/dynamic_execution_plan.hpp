// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DYNAMIC_EXECUTION_PLAN_HPP_
#define MSCCLPP_DYNAMIC_EXECUTION_PLAN_HPP_

#include <mscclpp/executor.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace mscclpp {

// Forward declaration
class Communicator;

/// Runtime parameters for dynamic execution plan
struct DynamicRuntimeParams {
  std::vector<int> peerRanks;                    ///< List of peer ranks
  std::vector<size_t> sendSizes;                ///< Send sizes per peer
  std::vector<size_t> recvSizes;                ///< Receive sizes per peer
  std::vector<size_t> sendOffsets;              ///< Send buffer offsets per peer
  std::vector<size_t> recvOffsets;              ///< Receive buffer offsets per peer
  size_t totalSendSize;                         ///< Total send buffer size
  size_t totalRecvSize;                         ///< Total receive buffer size
  int maxThreadBlocks;                          ///< Maximum thread blocks available
  size_t blockSize;                             ///< Thread block processing size
};

/// Variable substitution context for dynamic plans
struct VariableContext {
  std::unordered_map<std::string, std::string> variables;
  
  void setVariable(const std::string& name, const std::string& value) {
    variables[name] = value;
  }
  
  std::string substituteVariables(const std::string& template_str) const;
};

/// Dynamic operation template
struct DynamicOperationTemplate {
  std::string type;                             ///< Operation type (put, get, etc.)
  std::string inputChunk;                       ///< Input chunk variable
  std::string outputChunk;                      ///< Output chunk variable
  std::string peer;                             ///< Peer rank variable
  std::string channel;                          ///< Channel variable
  std::string threadblockCount;                 ///< Thread block count variable
  std::string size;                             ///< Size variable
  std::string step;                             ///< Step ID variable
};

/// Dynamic GPU template
struct DynamicGpuTemplate {
  int id;                                       ///< GPU ID
  std::string inputChunks;                      ///< Input chunks variable
  std::string outputChunks;                     ///< Output chunks variable
  int scratchChunks;                           ///< Scratch chunks count
  std::vector<DynamicOperationTemplate> operationTemplates;
};

/// Dynamic execution plan that can be instantiated at runtime
class DynamicExecutionPlan {
 public:
  /// Constructor
  /// @param planPath Path to the dynamic execution plan JSON file
  /// @param rank The rank of this process
  DynamicExecutionPlan(const std::string& planPath, int rank);
  
  /// Destructor
  ~DynamicExecutionPlan() = default;
  
  /// Instantiate the dynamic plan with runtime parameters
  /// @param params Runtime parameters for instantiation
  /// @return Concrete execution plan as JSON string
  std::string instantiate(const DynamicRuntimeParams& params);
  
  /// Create a concrete execution plan file for the given parameters
  /// @param params Runtime parameters for instantiation
  /// @param outputPath Path where to write the concrete plan
  /// @return Path to the created concrete plan file
  std::string createConcretePlan(const DynamicRuntimeParams& params, const std::string& outputPath);
  
  /// Get the collective name
  std::string collective() const { return collective_; }
  
  /// Get minimum message size
  size_t minMessageSize() const { return minMessageSize_; }
  
  /// Get maximum message size
  size_t maxMessageSize() const { return maxMessageSize_; }
  
  /// Check if this is a dynamic plan
  bool isDynamic() const { return isDynamic_; }
  
 private:
  // Fixed member order to match initialization order
  int rank_;                                    ///< Process rank
  std::string name_;                            ///< Plan name
  std::string collective_;                      ///< Collective operation type
  std::string protocol_;                        ///< Protocol type
  bool isDynamic_;                              ///< Whether this is a dynamic plan
  size_t minMessageSize_;                       ///< Minimum message size
  size_t maxMessageSize_;                       ///< Maximum message size
  int numThreadsPerBlock_;                      ///< Number of threads per block
  std::vector<DynamicGpuTemplate> gpuTemplates_; ///< GPU templates
  std::unordered_map<std::string, std::string> dynamicParams_; ///< Dynamic parameters
  
  /// Load dynamic plan from JSON
  void loadFromJson(const std::string& planPath);
  
  /// Calculate thread blocks needed for a given message size
  int calculateThreadBlocks(size_t messageSize) const;
};

/// Utility class for dynamic all-to-allv operations
class DynamicAllToAllv {
 public:
  /// Execute dynamic all-to-allv with runtime message sizes
  /// @param comm The communicator
  /// @param dynamicPlan The dynamic execution plan
  /// @param sendBuffer Send buffer
  /// @param recvBuffer Receive buffer
  /// @param sendSizes Send sizes per peer
  /// @param recvSizes Receive sizes per peer
  /// @param tag Operation tag
  /// @return True if successful, false otherwise
  static bool execute(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<DynamicExecutionPlan> dynamicPlan,
    void* sendBuffer, void* recvBuffer,
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes,
    int tag = 0);
  
  /// Create runtime parameters from send/recv sizes
  /// @param sendSizes Send sizes per peer
  /// @param recvSizes Receive sizes per peer
  /// @return Runtime parameters structure
  static DynamicRuntimeParams createRuntimeParams(
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes);
};

}  // namespace mscclpp

#endif  // MSCCLPP_DYNAMIC_EXECUTION_PLAN_HPP_