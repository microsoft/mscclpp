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
      throw std::runtime_error("[TorchCommMSCCLPP] " + collective_name + " unsupported reduce op type=" +
                               std::to_string(static_cast<int>(op.type())));
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

  // Install a topology-aware fallback selector. Same dispatcher used by
  // src/ext/nccl/nccl.cc — both backends pick up new algorithms automatically
  // as MSCCL++ adds them. Hardware capabilities are detected once.
  static const bool isNvlsSupported = mscclpp::isNvlsSupported();
  int major = 0, minor = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_.index()));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_.index()));
  static const std::pair<int, int> computeCapability = {major, minor};

  auto builder = mscclpp::collective::AlgorithmCollectionBuilder::getInstance();
  // TODO: This fallback selector duplicates the logic in src/ext/nccl/nccl.cc
  // (algoSelector) and src/ext/nccl/algorithm_selector.cc. The shared policy
  // (DSL match → topology-aware native selectors) should be promoted into a
  // single helper in src/ext/nccl/algorithm_selector.{hpp,cc} (e.g.
  // `defaultFallbackSelector`) and reused by both backends. Kept duplicated
  // here for now to keep this PR scoped to python/mscclpp_torchcomms/.
  // Shape mirrors upstream `algoSelector` (open-coded per-collective branches
  // rather than a local dispatch table) so the eventual extract-to-shared-
  // helper diff is mechanical.
  builder->setFallbackAlgorithmSelector(
      [](const auto& algoMapByCollective, const mscclpp::CollectiveRequest& request) {
        auto collectiveIt = algoMapByCollective.find(request.collective);
        if (collectiveIt == algoMapByCollective.end()) {
          return std::shared_ptr<mscclpp::Algorithm>{nullptr};
        }
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
            if (dslAlgo && mscclpp::nccl::matchExecutionPlan(dslAlgo, request)) {
              return algo;
            }
          }
        }

        // 2. Topology-aware native selectors
        if (request.nRanksPerNode != request.worldSize) {
          return mscclpp::nccl::selectMultiNodeAlgorithm(algoMap, request, config);
        }
        if (request.collective == "allgather") {
          return mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config);
        }
        if (request.collective == "allreduce") {
          return mscclpp::nccl::selectSingleNodeAllreduce(algoMap, request, config);
        }
        return std::shared_ptr<mscclpp::Algorithm>{nullptr};
      });
  algorithmCollection_ =
      builder->buildDefaultAlgorithms(reinterpret_cast<uintptr_t>(scratchBuffer_.get()), kScratchBufferSize,
                                      reinterpret_cast<uintptr_t>(flagBuffer_.get()), flagBufferSize_, rank_);

  event_pool_ = std::make_shared<MscclppGpuEventPool>(256);
  ncclFallback_ = NcclFallback::tryCreate(comm_, rank_, size_);

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

  ncclFallback_.reset();
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
// All collectives funnel through executeCollective(): build CollectiveRequest →
// AlgorithmCollection picks the algorithm (native or DSL) → execute via
// algo->execute() → wrap in TorchWorkMSCCLPP. New collectives MSCCL++ adds
// upstream become available here automatically; only collectives with no
// native algorithm AND a NCCL fallback path need a custom override (below).

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::executeCollective(const std::string& collective, const void* sendbuf,
                                                                  void* recvbuf, size_t sendBytes, size_t recvBytes,
                                                                  mscclpp::DataType dtype, mscclpp::ReduceOp reduceOp,
                                                                  bool async_op, std::chrono::milliseconds timeout) {
  std::unordered_map<std::string, std::vector<uint64_t>> hints;
  cudaStream_t stream = getOperationStream(async_op);
  mscclpp::CollectiveRequest request{size_,  nRanksPerNode_, rank_, sendbuf, recvbuf,
                                     sendBytes, stream,         collective, dtype,   hints};

  auto algo = algorithmCollection_.selectAlgorithm(request);
  if (!algo) {
    throw std::runtime_error("[TorchCommMSCCLPP] no algorithm registered for '" + collective + "' size=" +
                             std::to_string(sendBytes));
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
  return executeCollective("allreduce", tensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), tensor.nbytes(),
                           torchDtypeToMscclpp(tensor.scalar_type()),
                           torchReduceOpToMscclpp(op, "all_reduce"), async_op, options.timeout);
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
  mscclpp::CollectiveRequest request{size_,        nRanksPerNode_,    rank_,        input.data_ptr(), output.data_ptr(),
                                     static_cast<size_t>(input.nbytes()), stream, "reducescatter",  mscclppDtype, hints};
  auto algo = algorithmCollection_.selectAlgorithm(request);

  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), options.timeout, event_pool_);
  work->recordStart();
  if (algo) {
    algo->execute(comm_, input.data_ptr(), output.data_ptr(), input.nbytes(), output.nbytes(), mscclppDtype,
                  torchReduceOpToMscclpp(op, "reduce_scatter_single"), stream, executor_);
  } else if (ncclFallback_) {
    ncclFallback_->reduceScatter(input.data_ptr(), output.data_ptr(), output.numel(), dtype, op, stream);
  } else {
    throw std::runtime_error(
        "[TorchCommMSCCLPP] reduce_scatter_single: no MSCCL++ algorithm and no NCCL fallback (libnccl.so.2 not found)");
  }
  work->recordEnd();
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::broadcast(at::Tensor& tensor, int root, bool async_op,
                                                          const BroadcastOptions& options) {
  checkInitialized();
  if (!ncclFallback_)
    throw std::runtime_error("[TorchCommMSCCLPP] broadcast requires NCCL fallback (libnccl.so.2 not found)");
  TORCH_CHECK(tensor.is_contiguous(), "[TorchCommMSCCLPP] broadcast requires contiguous tensor");
  cudaStream_t stream = getOperationStream(async_op);
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), options.timeout, event_pool_);
  work->recordStart();
  ncclFallback_->broadcast(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), tensor.scalar_type(), root, stream);
  work->recordEnd();
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::barrier(bool async_op, const BarrierOptions& options) {
  checkInitialized();
  if (!ncclFallback_)
    throw std::runtime_error("[TorchCommMSCCLPP] barrier requires NCCL fallback (libnccl.so.2 not found)");
  cudaStream_t stream = getOperationStream(async_op);
  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), options.timeout, event_pool_);
  work->recordStart();
  ncclFallback_->barrier(stream);
  work->recordEnd();
  return work;
}

// --- Unsupported operations ---
//
// MSCCL++ focuses on bulk-synchronous data-parallel collectives. P2P, the
// tensor-list collective variants, scatter/gather, and split() aren't part
// of MSCCL++'s scope. Use a separate NCCL/RCCL TorchComm for these.

#define MSCCLPP_UNSUPPORTED(op, msg)                                                                       \
  throw std::runtime_error("[TorchCommMSCCLPP] " op " is not supported. " msg                              \
                           " Use a separate NCCL/RCCL TorchComm for this operation.")

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::send(const at::Tensor&, int, bool, const SendOptions&) {
  MSCCLPP_UNSUPPORTED("send()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::recv(at::Tensor&, int, bool, const RecvOptions&) {
  MSCCLPP_UNSUPPORTED("recv()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::batch_op_issue(const std::vector<BatchSendRecv::P2POp>&, bool,
                                                               const BatchP2POptions&) {
  MSCCLPP_UNSUPPORTED("batch_op_issue()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce(const at::Tensor&, int, const ReduceOp&, bool,
                                                       const ReduceOptions&) {
  MSCCLPP_UNSUPPORTED("reduce()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather(const std::vector<at::Tensor>&, const at::Tensor&, bool,
                                                           const AllGatherOptions&) {
  MSCCLPP_UNSUPPORTED("all_gather() (tensor-list variant)", "Use all_gather_single() instead.");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_v(const std::vector<at::Tensor>&, const at::Tensor&, bool,
                                                             const AllGatherOptions&) {
  MSCCLPP_UNSUPPORTED("all_gather_v()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter(at::Tensor&, const std::vector<at::Tensor>&,
                                                               const ReduceOp&, bool, const ReduceScatterOptions&) {
  MSCCLPP_UNSUPPORTED("reduce_scatter() (tensor-list variant)", "Use reduce_scatter_single() instead.");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_v(at::Tensor&, const std::vector<at::Tensor>&,
                                                                 const ReduceOp&, bool, const ReduceScatterOptions&) {
  MSCCLPP_UNSUPPORTED("reduce_scatter_v()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_v_single(at::Tensor&, const at::Tensor&,
                                                                    const std::vector<uint64_t>&,
                                                                    const std::vector<uint64_t>&, bool,
                                                                    const AllToAllvSingleOptions&) {
  MSCCLPP_UNSUPPORTED("all_to_all_v_single()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all(const std::vector<at::Tensor>&,
                                                           const std::vector<at::Tensor>&, bool,
                                                           const AllToAllOptions&) {
  MSCCLPP_UNSUPPORTED("all_to_all() (tensor-list variant)", "Use all_to_all_single() instead.");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::scatter(at::Tensor&, const std::vector<at::Tensor>&, int, bool,
                                                        const ScatterOptions&) {
  MSCCLPP_UNSUPPORTED("scatter()", "");
}
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::gather(const std::vector<at::Tensor>&, const at::Tensor&, int, bool,
                                                       const GatherOptions&) {
  MSCCLPP_UNSUPPORTED("gather()", "");
}
std::shared_ptr<TorchCommBackend> TorchCommMSCCLPP::split(const std::vector<int>&, const std::string&,
                                                          const CommOptions&) {
  MSCCLPP_UNSUPPORTED("split()", "");
}

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
