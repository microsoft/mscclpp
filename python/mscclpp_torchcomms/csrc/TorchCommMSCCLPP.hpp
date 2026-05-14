// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/ATen.h>

#include <comms/torchcomms/TorchCommBackend.hpp>
#include <memory>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/gpu.hpp>
#include <string>
#include <string_view>

#include "TorchWorkMSCCLPP.hpp"

namespace torch::comms {

/// MSCCL++ backend for TorchComms.
///
/// This is a thin adapter that maps TorchCommBackend collective operations to
/// MSCCL++'s AlgorithmCollection. Algorithm selection (native vs DSL, which
/// variant for a given message size / topology) is handled entirely by MSCCL++
/// via AlgorithmCollection::selectAlgorithm(). The backend just builds a
/// CollectiveRequest and calls algo->execute().
///
/// Supported collectives: all_reduce, all_gather_single, reduce_scatter_single,
/// all_to_all_single. All others throw with guidance to use NCCL/RCCL.
///
/// Lifecycle:
///   1. TorchCommFactory creates an instance via the registered "mscclpp" factory
///   2. Caller invokes init() with device, name, and CommOptions (including c10d::Store)
///   3. init() bootstraps rank discovery, creates the MSCCL++ Communicator, builds
///      the AlgorithmCollection with all default native + DSL algorithms
///   4. Collectives are dispatched through executeCollective()
///   5. finalize() syncs streams, runs a bootstrap barrier, and tears down in reverse order
class TorchCommMSCCLPP : public TorchCommBackend, public std::enable_shared_from_this<TorchCommMSCCLPP> {
 public:
  static constexpr std::string_view kBackendName = "mscclpp";

  TorchCommMSCCLPP();
  ~TorchCommMSCCLPP() override;

  TorchCommMSCCLPP(const TorchCommMSCCLPP&) = delete;
  TorchCommMSCCLPP& operator=(const TorchCommMSCCLPP&) = delete;

  // Lifecycle
  void init(at::Device device, const std::string& name, const CommOptions& options = {}) override;
  void finalize() override;

  // Metadata
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;
  const CommOptions& getOptions() const override;
  const at::Device& getDevice() const override;

  // Point-to-point (unsupported)
  c10::intrusive_ptr<TorchWork> send(const at::Tensor& tensor, int dst, bool async_op,
                                     const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(at::Tensor& tensor, int src, bool async_op,
                                     const RecvOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> batch_op_issue(const std::vector<BatchSendRecv::P2POp>& ops, bool async_op,
                                               const BatchP2POptions& options = {}) override;

  // Collectives
  c10::intrusive_ptr<TorchWork> broadcast(at::Tensor& tensor, int root, bool async_op,
                                          const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(at::Tensor& tensor, const ReduceOp& op, bool async_op,
                                           const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(const at::Tensor& tensor, int root, const ReduceOp& op, bool async_op,
                                       const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(const std::vector<at::Tensor>& tensor_list, const at::Tensor& tensor,
                                           bool async_op, const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(const std::vector<at::Tensor>& tensor_list, const at::Tensor& tensor,
                                             bool async_op, const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(at::Tensor& output, const at::Tensor& input, bool async_op,
                                                  const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(at::Tensor& output, const std::vector<at::Tensor>& input_list,
                                               const ReduceOp& op, bool async_op,
                                               const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(at::Tensor& output, const std::vector<at::Tensor>& input_list,
                                                 const ReduceOp& op, bool async_op,
                                                 const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(at::Tensor& output, const at::Tensor& input, const ReduceOp& op,
                                                      bool async_op,
                                                      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(at::Tensor& output, const at::Tensor& input, bool async_op,
                                                  const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(at::Tensor& output, const at::Tensor& input,
                                                    const std::vector<uint64_t>& output_split_sizes,
                                                    const std::vector<uint64_t>& input_split_sizes, bool async_op,
                                                    const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(const std::vector<at::Tensor>& output_tensor_list,
                                           const std::vector<at::Tensor>& input_tensor_list, bool async_op,
                                           const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(bool async_op, const BarrierOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> scatter(at::Tensor& output_tensor, const std::vector<at::Tensor>& input_tensor_list,
                                        int root, bool async_op, const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(const std::vector<at::Tensor>& output_tensor_list,
                                       const at::Tensor& input_tensor, int root, bool async_op,
                                       const GatherOptions& options = {}) override;

  // Communicator management (unsupported)
  std::shared_ptr<TorchCommBackend> split(const std::vector<int>& ranks, const std::string& name,
                                          const CommOptions& options = {}) override;

 private:
  void checkInitialized() const;

  /// Map PyTorch scalar type to MSCCL++ DataType.
  static mscclpp::DataType torchDtypeToMscclpp(at::ScalarType dtype);

  /// Map TorchComms ReduceOp to MSCCL++ ReduceOp.
  /// Throws if the op is not supported by MSCCL++ native kernels.
  static mscclpp::ReduceOp torchReduceOpToMscclpp(const ReduceOp& op, const std::string& collective_name);

  /// Get the appropriate stream for an operation.
  cudaStream_t getOperationStream(bool async_op) const;

  /// Central dispatch for all supported collectives.
  ///
  /// Builds a CollectiveRequest from the arguments, asks AlgorithmCollection to
  /// select the best algorithm (native or DSL), creates a TorchWorkMSCCLPP handle
  /// with start/end GPU events, executes the algorithm, and returns the work handle.
  /// The caller's stream waits on the end event when work->wait() is called.
  c10::intrusive_ptr<TorchWork> executeCollective(const std::string& collective, const void* sendbuf, void* recvbuf,
                                                  size_t sendBytes, size_t recvBytes, mscclpp::DataType dtype,
                                                  mscclpp::ReduceOp reduceOp, bool async_op,
                                                  std::chrono::milliseconds timeout);

  bool initialized_ = false;
  at::Device device_{at::kCUDA};
  std::string name_;
  CommOptions options_;
  int rank_ = 0;
  int size_ = 1;
  int nRanksPerNode_ = 1;  // cached from bootstrap; used in CollectiveRequest for algorithm selection

  /// MSCCL++ communicator — owns the bootstrap, context, and all registered connections.
  std::shared_ptr<mscclpp::Communicator> comm_;

  /// Executor for DSL-based algorithms. Native algorithms ignore this, but DSL
  /// algorithms need it to interpret JSON execution plans. Always passed to
  /// algo->execute() so the backend doesn't need to distinguish algorithm types.
  std::shared_ptr<mscclpp::Executor> executor_;

  /// Registry of all available algorithms (native + DSL). Built once in init()
  /// via AlgorithmCollectionBuilder::buildDefaultAlgorithms(). selectAlgorithm()
  /// picks the best algorithm for a given collective + message size + topology.
  mscclpp::AlgorithmCollection algorithmCollection_;

  /// Dedicated stream for async collective launches. Sync ops use the caller's
  /// current PyTorch CUDA stream instead, so the kernel is inline with their work.
  cudaStream_t internal_stream_ = nullptr;

  /// Reusable GPU event pool shared across all TorchWorkMSCCLPP handles from
  /// this communicator. Avoids cudaEventCreate/Destroy overhead per collective.
  std::shared_ptr<MscclppGpuEventPool> event_pool_;

  /// GPU scratch memory used by native algorithms (e.g., allreduce RS+AG pipeline)
  /// for intermediate results. 128MB is the default size matching MSCCL++ conventions.
  /// Allocated via GpuBuffer (cuMemMap) so POSIX file descriptors are registered
  /// in the unix socket server for cross-rank IPC sharing.
  std::shared_ptr<char> scratchBuffer_;
  static constexpr size_t kScratchBufferSize = 1 << 27;  // 128MB

  /// Flag buffer shared pointer — must be kept alive for the lifetime of the
  /// communicator since AlgorithmCollection references it.
  std::shared_ptr<void> flagBuffer_;
  size_t flagBufferSize_ = 0;
};

}  // namespace torch::comms
