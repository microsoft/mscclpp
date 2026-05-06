// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "TorchCommMSCCLPP.hpp"

#include <ATen/cuda/CUDAContext.h>

#include <comms/torchcomms/TorchCommFactory.hpp>
#include <cstdlib>
#include <iostream>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>
#include <stdexcept>

#include "NcclFallback.hpp"
#include "TorchCommMSCCLPPBootstrap.hpp"
#include "algorithm_selector.hpp"  // shared with src/ext/nccl

namespace torch::comms {

// --- Helpers ---

mscclpp::DataType TorchCommMSCCLPP::torchDtypeToMscclpp(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return mscclpp::DataType::FLOAT32;
    case at::kHalf:
      return mscclpp::DataType::FLOAT16;
    case at::kBFloat16:
      return mscclpp::DataType::BFLOAT16;
    case at::kInt:
      return mscclpp::DataType::INT32;
    case at::kUInt32:
      return mscclpp::DataType::UINT32;
    default:
      throw std::runtime_error("[TorchCommMSCCLPP] unsupported dtype: " + std::string(at::toString(dtype)));
  }
}

mscclpp::ReduceOp TorchCommMSCCLPP::torchReduceOpToMscclpp(const ReduceOp& op, const std::string& collective_name) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
    // FSDP2 sends PREMUL_SUM (with the divide factor pre-applied to the gradient)
    // and AVG for reduce_scatter. MSCCL++ kernels only implement SUM, but the
    // caller has already done the scaling for PREMUL_SUM, and FSDP2's
    // set_gradient_divide_factor(1.0) makes AVG behave as SUM.
    case ReduceOp::RedOpType::PREMUL_SUM:
    case ReduceOp::RedOpType::AVG:
      return mscclpp::SUM;
    case ReduceOp::RedOpType::MIN:
      return mscclpp::MIN;
    default:
      throw std::runtime_error("[TorchCommMSCCLPP] " + collective_name +
                               " unsupported reduce op type=" + std::to_string(static_cast<int>(op.type())));
  }
}

// Async ops use the dedicated internal stream; sync ops use the caller's
// current PyTorch CUDA stream so the launch is ordered inline with their work.
cudaStream_t TorchCommMSCCLPP::getOperationStream(bool async_op) const {
  return async_op ? internal_stream_ : at::cuda::getCurrentCUDAStream(device_.index()).stream();
}

void TorchCommMSCCLPP::checkInitialized() const {
  if (!initialized_) throw std::runtime_error("[TorchCommMSCCLPP] not initialized; call init() first");
}

// Wraps a fallback dispatch in start/end GPU events on `stream`. Throws if
// libnccl could not be dlopen'd at init time (fallback unavailable).
template <typename Fn>
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::ncclFallback(const char* op, cudaStream_t stream,
                                                             std::chrono::milliseconds timeout, Fn&& body) {
  if (!nccl_) {
    throw std::runtime_error(std::string("[TorchCommMSCCLPP] ") + op +
                             " requires NCCL fallback (libnccl.so.2 not found)");
  }
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), timeout, event_pool_);
  work->recordStart();
  std::forward<Fn>(body)();
  work->recordEnd();
  return work;
}

std::shared_ptr<mscclpp::Algorithm> TorchCommMSCCLPP::selectAlgorithm(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::shared_ptr<mscclpp::Algorithm>>>& algoMapByCollective,
    const mscclpp::CollectiveRequest& request) {
  // Hardware capabilities are detected once on first call (per process).
  static const bool isNvlsSupported = mscclpp::isNvlsSupported();
  static const std::pair<int, int> computeCapability = []() {
    int dev = 0;
    cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return std::make_pair(major, minor);
  }();

  auto collectiveIt = algoMapByCollective.find(request.collective);
  if (collectiveIt == algoMapByCollective.end()) return nullptr;
  const auto& algoMap = collectiveIt->second;

  const bool isCuMemMap = mscclpp::isCuMemMapAllocated(const_cast<void*>(request.inputBuffer)) &&
                          mscclpp::isCuMemMapAllocated(request.outputBuffer);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(request.stream, &capture);
  mscclpp::nccl::AlgorithmSelectorConfig config{
      .symmetricMemory = false,
      .nvlsSupported = isNvlsSupported,
      .isCuMemMapAllocated = isCuMemMap,
      .inCaptureMode = (capture == cudaStreamCaptureStatusActive),
      .computeCapability = computeCapability,
      .ncclDlopenSharedLib = false,
  };

  // 1. DSL execution plans
  for (const auto& [name, algo] : algoMap) {
    (void)name;
    if (algo->type() == mscclpp::AlgorithmType::DSL) {
      auto dslAlgo = std::dynamic_pointer_cast<mscclpp::DslAlgorithm>(algo);
      if (dslAlgo && mscclpp::nccl::matchExecutionPlan(dslAlgo, request)) return algo;
    }
  }

  // 2. Topology-aware native selectors
  if (request.nRanksPerNode != request.worldSize) {
    return mscclpp::nccl::selectMultiNodeAlgorithm(algoMap, request, config);
  }
  if (request.collective == "allgather") return mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config);
  if (request.collective == "allreduce") return mscclpp::nccl::selectSingleNodeAllreduce(algoMap, request, config);
  return nullptr;
}

namespace {
// Reduce ops that map cleanly to MSCCL++ kernels (SUM-family + MIN).
// Anything else (MAX, PRODUCT, ...) goes through NcclFallback.
bool isMscclppNativeReduceOp(const ReduceOp& op) {
  using T = ReduceOp::RedOpType;
  switch (op.type()) {
    case T::SUM:
    case T::PREMUL_SUM:
    case T::AVG:
    case T::MIN:
      return true;
    default:
      return false;
  }
}
}  // namespace

// --- Lifecycle ---

TorchCommMSCCLPP::TorchCommMSCCLPP() = default;

TorchCommMSCCLPP::~TorchCommMSCCLPP() {
  if (initialized_) {
    try {
      finalize();
    } catch (...) {
    }
  }
}

void TorchCommMSCCLPP::init(at::Device device, const std::string& name, const CommOptions& options) {
  if (initialized_) throw std::runtime_error("[TorchCommMSCCLPP] already initialized");

  device_ = device;
  name_ = name;
  options_ = options;

  // Bootstrap + communicator
  auto bootstrap = std::make_unique<TorchCommMSCCLPPBootstrap>(options.store, device, options.timeout);
  rank_ = bootstrap->getRank();
  size_ = bootstrap->getSize();
  comm_ = bootstrap->createCommunicator(name, options);

  mscclpp::CudaDeviceGuard deviceGuard(device_.index());
  nRanksPerNode_ = comm_->bootstrap()->getNranksPerNode();

  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&internal_stream_, cudaStreamNonBlocking));

  // Scratch buffer must use GpuBuffer (cuMemMap) so its POSIX fd is registered
  // in the unix socket server; plain cudaMalloc causes "Requested fd not found"
  // crashes during cross-rank IPC sharing.
  scratchBuffer_ = mscclpp::GpuBuffer<char>(kScratchBufferSize).memory();
  executor_ = std::make_shared<mscclpp::Executor>(comm_, scratchBuffer_);

  auto [flagBuf, flagSize] = mscclpp::getFlagBuffer();
  flagBuffer_ = flagBuf;
  flagBufferSize_ = flagSize;

  // Install the topology-aware fallback selector. Body is out-of-line in
  // selectAlgorithm() to keep init() readable.
  //
  // TODO: This selector duplicates src/ext/nccl/nccl.cc::algoSelector. The
  // shared policy (DSL match → topology-aware native selectors) should be
  // promoted into a single helper in src/ext/nccl/algorithm_selector.{hpp,cc}
  // (e.g. `defaultFallbackSelector`) and reused by both backends. Kept
  // duplicated here for now to scope this PR to python/mscclpp_torchcomms/.
  auto builder = mscclpp::collective::AlgorithmCollectionBuilder::getInstance();
  builder->setFallbackAlgorithmSelector(&TorchCommMSCCLPP::selectAlgorithm);
  algorithmCollection_ =
      builder->buildDefaultAlgorithms(reinterpret_cast<uintptr_t>(scratchBuffer_.get()), kScratchBufferSize,
                                      reinterpret_cast<uintptr_t>(flagBuffer_.get()), flagBufferSize_, rank_);

  event_pool_ = std::make_shared<MscclppGpuEventPool>(256);
  nccl_ = NcclFallback::tryCreate(comm_, rank_, size_);

  initialized_ = true;
}

void TorchCommMSCCLPP::finalize() {
  if (!initialized_) return;

  // Drain our streams while NVLink memory is alive, then bootstrap-barrier so
  // no rank tears down its communicator while another's kernel is still
  // polling its NVLink memory.
  if (internal_stream_) MSCCLPP_CUDATHROW(cudaStreamSynchronize(internal_stream_));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(device_.index()).stream()));
  comm_->bootstrap()->barrier();

  nccl_.reset();
  executor_.reset();
  event_pool_.reset();
  if (internal_stream_) {
    MSCCLPP_CUDATHROW(cudaStreamDestroy(internal_stream_));
    internal_stream_ = nullptr;
  }
  scratchBuffer_.reset();
  flagBuffer_.reset();
  comm_.reset();
  initialized_ = false;
}

// --- Metadata ---

int TorchCommMSCCLPP::getRank() const { return rank_; }
int TorchCommMSCCLPP::getSize() const { return size_; }
std::string_view TorchCommMSCCLPP::getBackendName() const { return kBackendName; }
std::string_view TorchCommMSCCLPP::getCommName() const { return name_; }
const CommOptions& TorchCommMSCCLPP::getOptions() const { return options_; }
const at::Device& TorchCommMSCCLPP::getDevice() const { return device_; }

// --- Collective dispatch ---
//
// All collectives funnel through executeCollective() (when MSCCL++ has a
// native algorithm) or ncclFallback() (when it doesn't). Each method below
// has one of three shapes:
//
//   1. NATIVE-ONLY    — body is just `return executeCollective(...);`
//                       MSCCL++ has algorithms for the entire collective.
//                       Examples: all_gather_single, all_to_all_single
//
//   2. FALLBACK-ONLY  — body is just `return ncclFallback(...);`
//                       No MSCCL++ native algorithm exists.
//                       To migrate to native once MSCCL++ adds support:
//                       replace `return ncclFallback(...)` with
//                       `return executeCollective(...)`.
//                       Examples: broadcast, barrier, send, recv, reduce,
//                                 all_to_all_v_single
//
//   3. NATIVE+FALLBACK — try MSCCL++ first, fall back to NCCL when no
//                       native algorithm matches the request (e.g. unusual
//                       reduce op, message size with no plan).
//                       To remove the fallback once MSCCL++ covers the gap:
//                       delete the `ncclFallback(...)` call and any guards
//                       that gated it.
//                       Examples: all_reduce, reduce_scatter_single

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::executeCollective(const std::string& collective, const void* sendbuf,
                                                                  void* recvbuf, size_t sendBytes, size_t recvBytes,
                                                                  mscclpp::DataType dtype, mscclpp::ReduceOp reduceOp,
                                                                  bool async_op, std::chrono::milliseconds timeout) {
  std::unordered_map<std::string, std::vector<uint64_t>> hints;
  cudaStream_t stream = getOperationStream(async_op);
  mscclpp::CollectiveRequest request{size_,     nRanksPerNode_, rank_,      sendbuf, recvbuf,
                                     sendBytes, stream,         collective, dtype,   hints};

  auto algo = algorithmCollection_.selectAlgorithm(request);
  if (!algo) {
    throw std::runtime_error("[TorchCommMSCCLPP] no algorithm registered for '" + collective +
                             "' size=" + std::to_string(sendBytes));
  }

  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), timeout, event_pool_);
  work->recordStart();
  algo->execute(comm_, sendbuf, recvbuf, sendBytes, recvBytes, dtype, reduceOp, stream, executor_);
  work->recordEnd();
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_reduce(at::Tensor& tensor, const ReduceOp& op, bool async_op,
                                                           const AllReduceOptions& options) {
  checkInitialized();
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] all_reduce requires contiguous tensor");
  if (isMscclppNativeReduceOp(op)) {
    return executeCollective("allreduce", tensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), tensor.nbytes(),
                             torchDtypeToMscclpp(tensor.scalar_type()), torchReduceOpToMscclpp(op, "all_reduce"),
                             async_op, options.timeout);
  }
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("all_reduce", stream, options.timeout, [&] {
    nccl_->allReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), op, stream);
  });
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_single(at::Tensor& output, const at::Tensor& input,
                                                                  bool async_op,
                                                                  const AllGatherSingleOptions& options) {
  checkInitialized();
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "[TorchCommMSCCLPP] all_gather_single requires contiguous tensors");
  return executeCollective("allgather", input.data_ptr(), output.data_ptr(), input.nbytes(), output.nbytes(),
                           torchDtypeToMscclpp(input.scalar_type()), mscclpp::NOP, async_op, options.timeout);
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_single(at::Tensor& output, const at::Tensor& input,
                                                                  bool async_op, const AllToAllSingleOptions& options) {
  checkInitialized();
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "[TorchCommMSCCLPP] all_to_all_single requires contiguous tensors");
  return executeCollective("alltoall", input.data_ptr(), output.data_ptr(), input.nbytes(), output.nbytes(),
                           torchDtypeToMscclpp(input.scalar_type()), mscclpp::NOP, async_op, options.timeout);
}

// reduce_scatter: try MSCCL++ first; fall back to NCCL if no native algorithm.
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_single(at::Tensor& output, const at::Tensor& input,
                                                                      const ReduceOp& op, bool async_op,
                                                                      const ReduceScatterSingleOptions& options) {
  checkInitialized();
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "[TorchCommMSCCLPP] reduce_scatter_single requires contiguous tensors");

  const auto dtype = input.scalar_type();
  const auto mscclppDtype = torchDtypeToMscclpp(dtype);
  cudaStream_t stream = getOperationStream(async_op);

  std::unordered_map<std::string, std::vector<uint64_t>> hints;
  mscclpp::CollectiveRequest request{size_,
                                     nRanksPerNode_,
                                     rank_,
                                     input.data_ptr(),
                                     output.data_ptr(),
                                     static_cast<size_t>(input.nbytes()),
                                     stream,
                                     "reducescatter",
                                     mscclppDtype,
                                     hints};
  auto algo = isMscclppNativeReduceOp(op) ? algorithmCollection_.selectAlgorithm(request) : nullptr;
  if (algo) {
    auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), options.timeout, event_pool_);
    work->recordStart();
    algo->execute(comm_, input.data_ptr(), output.data_ptr(), input.nbytes(), output.nbytes(), mscclppDtype,
                  torchReduceOpToMscclpp(op, "reduce_scatter_single"), stream, executor_);
    work->recordEnd();
    return work;
  }
  return ncclFallback("reduce_scatter_single", stream, options.timeout, [&] {
    nccl_->reduceScatter(input.data_ptr(), output.data_ptr(), output.numel(), dtype, op, stream);
  });
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::broadcast(at::Tensor& tensor, int root, bool async_op,
                                                          const BroadcastOptions& options) {
  checkInitialized();
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] broadcast requires contiguous tensor");
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("broadcast", stream, options.timeout, [&] {
    nccl_->broadcast(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), root, stream);
  });
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::barrier(bool async_op, const BarrierOptions& options) {
  checkInitialized();
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("barrier", stream, options.timeout, [&] { nccl_->barrier(stream); });
}

// --- Unsupported operations ---
//
// MSCCL++ focuses on bulk-synchronous data-parallel collectives. P2P, the
// tensor-list collective variants, scatter/gather, and split() aren't part
// of MSCCL++'s scope. Use a separate NCCL/RCCL TorchComm for these.

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::send(const at::Tensor& tensor, int peer, bool async_op,
                                                     const SendOptions& options) {
  checkInitialized();
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] send requires contiguous tensor");
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("send", stream, options.timeout, [&] {
    nccl_->send(tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), peer, stream);
  });
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::recv(at::Tensor& tensor, int peer, bool async_op,
                                                     const RecvOptions& options) {
  checkInitialized();
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] recv requires contiguous tensor");
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("recv", stream, options.timeout, [&] {
    nccl_->recv(tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), peer, stream);
  });
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce(const at::Tensor& tensor, int root, const ReduceOp& op,
                                                       bool async_op, const ReduceOptions& options) {
  checkInitialized();
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] reduce requires contiguous tensor");
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("reduce", stream, options.timeout, [&] {
    nccl_->reduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), op, root, stream);
  });
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_v_single(at::Tensor& output, const at::Tensor& input,
                                                                    const std::vector<uint64_t>& outputSplitSizes,
                                                                    const std::vector<uint64_t>& inputSplitSizes,
                                                                    bool async_op,
                                                                    const AllToAllvSingleOptions& options) {
  checkInitialized();
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "[TorchCommMSCCLPP] all_to_all_v_single requires contiguous tensors");
  TORCH_CHECK(static_cast<int>(inputSplitSizes.size()) == size_ && static_cast<int>(outputSplitSizes.size()) == size_,
              "[TorchCommMSCCLPP] all_to_all_v_single: split-size vectors must have length world_size");
  std::vector<uint64_t> sendOffsets(size_, 0), recvOffsets(size_, 0);
  for (int i = 1; i < size_; ++i) {
    sendOffsets[i] = sendOffsets[i - 1] + inputSplitSizes[i - 1];
    recvOffsets[i] = recvOffsets[i - 1] + outputSplitSizes[i - 1];
  }
  cudaStream_t stream = getOperationStream(async_op);
  return ncclFallback("all_to_all_v_single", stream, options.timeout, [&] {
    nccl_->allToAllV(input.data_ptr(), output.data_ptr(), inputSplitSizes, outputSplitSizes, sendOffsets,
                             recvOffsets, input.scalar_type(), stream);
  });
}

#define MSCCLPP_UNSUPPORTED(op, msg)                                          \
  throw std::runtime_error("[TorchCommMSCCLPP] " op " is not supported. " msg \
                           " Use a separate NCCL/RCCL TorchComm for this operation.")

// One-liner stub for unsupported collectives that return a TorchWork handle.
#define UNSUPPORTED_OP(method, signature, label, msg)                \
  c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::method signature { \
    MSCCLPP_UNSUPPORTED(label, msg);                                 \
  }

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::batch_op_issue(const std::vector<BatchSendRecv::P2POp>&, bool,
                                                               const BatchP2POptions&) {
  MSCCLPP_UNSUPPORTED("batch_op_issue()", "");
}

UNSUPPORTED_OP(all_gather,
               (const std::vector<at::Tensor>&, const at::Tensor&, bool, const AllGatherOptions&),
               "all_gather() (tensor-list variant)", "Use all_gather_single() instead.")
UNSUPPORTED_OP(all_gather_v,
               (const std::vector<at::Tensor>&, const at::Tensor&, bool, const AllGatherOptions&),
               "all_gather_v()", "")
UNSUPPORTED_OP(reduce_scatter,
               (at::Tensor&, const std::vector<at::Tensor>&, const ReduceOp&, bool, const ReduceScatterOptions&),
               "reduce_scatter() (tensor-list variant)", "Use reduce_scatter_single() instead.")
UNSUPPORTED_OP(reduce_scatter_v,
               (at::Tensor&, const std::vector<at::Tensor>&, const ReduceOp&, bool, const ReduceScatterOptions&),
               "reduce_scatter_v()", "")
UNSUPPORTED_OP(all_to_all,
               (const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, bool, const AllToAllOptions&),
               "all_to_all() (tensor-list variant)", "Use all_to_all_single() instead.")
UNSUPPORTED_OP(scatter,
               (at::Tensor&, const std::vector<at::Tensor>&, int, bool, const ScatterOptions&),
               "scatter()", "")
UNSUPPORTED_OP(gather,
               (const std::vector<at::Tensor>&, const at::Tensor&, int, bool, const GatherOptions&),
               "gather()", "")

std::shared_ptr<TorchCommBackend> TorchCommMSCCLPP::split(const std::vector<int>&, const std::string&,
                                                          const CommOptions&) {
  MSCCLPP_UNSUPPORTED("split()", "");
}

#undef UNSUPPORTED_OP
#undef MSCCLPP_UNSUPPORTED

// --- Factory registration ---

namespace {
struct Registration {
  Registration() {
    TorchCommFactory::get().register_backend("mscclpp", []() { return std::make_shared<TorchCommMSCCLPP>(); });
  }
};
static const Registration registration{};
}  // namespace

}  // namespace torch::comms
