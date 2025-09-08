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
  int num_ranks;                                ///< Number of ranks
  std::vector<size_t> send_sizes;              ///< Send sizes per peer
  std::vector<size_t> recv_sizes;              ///< Receive sizes per peer
  std::vector<size_t> send_offsets;            ///< Send buffer offsets per peer
  std::vector<size_t> recv_offsets;            ///< Receive buffer offsets per peer
  std::vector<int> peerRanks;                  ///< List of peer ranks (for compatibility)
  size_t totalSendSize;                        ///< Total send buffer size
  size_t totalRecvSize;                        ///< Total receive buffer size
  int maxThreadBlocks;                         ///< Maximum thread blocks available
  size_t blockSize;                            ///< Thread block processing size
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

// Forward declaration
class DynamicExecutionPlan;

/// Utility class for dynamic all-to-allv operations
class DynamicAllToAllv {
 public:
  /// Constructor
  /// @param plan Reference to the dynamic execution plan
  DynamicAllToAllv(DynamicExecutionPlan& plan);
  
  /// Execute dynamic all-to-allv with runtime message sizes using MSCCLPP execution engine
  /// @param send_buff Send buffer
  /// @param send_sizes Send sizes per peer
  /// @param send_offsets Send buffer offsets per peer
  /// @param recv_buff Receive buffer
  /// @param recv_sizes Receive sizes per peer
  /// @param recv_offsets Receive buffer offsets per peer
  /// @param comm The communicator
  /// @param executor The MSCCLPP executor
  /// @param stream CUDA stream
  void execute(
    void* send_buff,
    const std::vector<size_t>& send_sizes,
    const std::vector<size_t>& send_offsets,
    void* recv_buff,
    const std::vector<size_t>& recv_sizes,
    const std::vector<size_t>& recv_offsets,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<Executor> executor,
    cudaStream_t stream);

  /// Create runtime parameters from send/recv sizes (static method for compatibility)
  /// @param sendSizes Send sizes per peer
  /// @param recvSizes Receive sizes per peer
  /// @return Runtime parameters structure
  static DynamicRuntimeParams createRuntimeParams(
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes);

  /// Execute method for compatibility (static)
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

 private:
  DynamicExecutionPlan& plan_;
  int rank_;
};

/// Dynamic execution plan that can be instantiated at runtime
class DynamicExecutionPlan {
 public:
  /// Constructor
  /// @param planPath Path to the dynamic execution plan JSON file
  /// @param rank The rank of this process
  DynamicExecutionPlan(const std::string& planPath, int rank);
  
  /// Destructor
  ~DynamicExecutionPlan();
  
  /// Instantiate the dynamic plan with runtime parameters
  /// @param params Runtime parameters for instantiation
  /// @return Concrete execution plan as JSON string
  std::string instantiate(const DynamicRuntimeParams& params);
  
  /// Create a concrete ExecutionPlan object for the given parameters
  /// @param params Runtime parameters for instantiation
  /// @return Shared pointer to concrete ExecutionPlan
  std::shared_ptr<ExecutionPlan> createExecutionPlan(const DynamicRuntimeParams& params);
  
  /// Create a DynamicAllToAllv object
  /// @return Unique pointer to DynamicAllToAllv
  std::unique_ptr<DynamicAllToAllv> createAllToAllv();
  
  /// Create a concrete execution plan file for the given parameters (for compatibility)
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
  
  /// Get the rank
  int getRank() const { return rank_; }
  
  /// Clean up temporary files created by this plan
  void cleanup();

 private:
  void loadFromJson(const std::string& planPath);
  int calculateThreadBlocks(size_t messageSize) const;
  std::string createLocalCopyVersion(const DynamicRuntimeParams& params, 
                                   const VariableContext& var_context);
  
  // Forward declare a JsonType to avoid exposing nlohmann::json in header
  class JsonType;
  void processJsonTemplateVariables(JsonType& json_obj, const VariableContext& var_context);
  void updateOperationWithRuntimeParams(JsonType& op, 
                                       const DynamicRuntimeParams& params,
                                       const VariableContext& var_context);
  void processOperationTemplates(JsonType& gpu_json, 
                                const DynamicRuntimeParams& params,
                                const VariableContext& var_context);
  void substituteOperationTemplateVariables(JsonType& operation_template,
                                           const DynamicRuntimeParams& params,
                                           const VariableContext& var_context);

  int rank_;                                         ///< Current rank
  std::string name_;                                 ///< Plan name
  std::string collective_;                           ///< Collective operation name
  std::string protocol_;                             ///< Protocol name
  bool isDynamic_;                                   ///< Whether this is a dynamic plan
  size_t minMessageSize_;                            ///< Minimum message size
  size_t maxMessageSize_;                            ///< Maximum message size
  int numThreadsPerBlock_;                           ///< Number of threads per block
  std::unordered_map<std::string, std::string> dynamicParams_;    ///< Dynamic parameters
  std::vector<DynamicGpuTemplate> gpuTemplates_;     ///< GPU templates
  std::string temp_file_path_;                       ///< Path to temporary file (for cleanup)
  
  // Use a pointer to avoid including nlohmann/json.hpp in header
  std::unique_ptr<JsonType> templateJson_;     ///< Original template JSON from DSL
};

}  // namespace mscclpp

#endif  // MSCCLPP_DYNAMIC_EXECUTION_PLAN_HPP_