// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_
#define MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_

#include <memory>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <unordered_map>

namespace mscclpp {
namespace nccl {

/// Configuration for algorithm selection
struct AlgorithmSelectorConfig {
  bool symmetricMemory;
  bool nvlsSupported;
  bool isCuMemMapAllocated;
  bool inCaptureMode;
  std::pair<int, int> computeCapability;
  bool ncclDlopenSharedLib;
};

/// Select an algorithm for single-node allreduce on Blackwell architecture
std::shared_ptr<Algorithm> selectSingleNodeAllreduceBlackwell(
    const std::unordered_map<std::string, std::shared_ptr<Algorithm>>& algoMap, const CollectiveRequest& request,
    const AlgorithmSelectorConfig& config);

/// Select an algorithm for single-node allreduce on non-Blackwell architectures
std::shared_ptr<Algorithm> selectSingleNodeAllreduce(
    const std::unordered_map<std::string, std::shared_ptr<Algorithm>>& algoMap, const CollectiveRequest& request,
    const AlgorithmSelectorConfig& config);

/// Select an algorithm for single-node allgather
std::shared_ptr<Algorithm> selectSingleNodeAllgather(
    const std::unordered_map<std::string, std::shared_ptr<Algorithm>>& algoMap, const CollectiveRequest& request,
    const AlgorithmSelectorConfig& config);

/// Select an algorithm for multi-node collective operations
/// Currently returns nullptr to fallback to NCCL/RCCL
/// TODO: Implement multi-node NVLS and multi-node IB algorithms
std::shared_ptr<Algorithm> selectMultiNodeAlgorithm(
    const std::unordered_map<std::string, std::shared_ptr<Algorithm>>& algoMap, const CollectiveRequest& request,
    const AlgorithmSelectorConfig& config);

/// Check if an execution plan matches the request
bool matchExecutionPlan(std::shared_ptr<DslAlgorithm> algo, const CollectiveRequest& request);

}  // namespace nccl
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_NCCL_ALGORITHM_SELECTOR_HPP_
