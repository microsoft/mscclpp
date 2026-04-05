// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "TorchCommMSCCLPP.hpp"

#include <ATen/cuda/CUDAContext.h>

#include <comms/torchcomms/TorchCommFactory.hpp>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>
#include <stdexcept>

#include "TorchCommMSCCLPPBootstrap.hpp"

// Use the same algorithm selector as the NCCL extension — it has proper
// topology-aware selection logic for message size, NVLS, compute capability, etc.
#include "algorithm_selector.hpp"

namespace torch::comms {

// --- Helpers ---

// Maps PyTorch tensor dtypes to MSCCL++ DataType enum values.
// Only types supported by MSCCL++ kernels are mapped; others throw.
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
      throw std::runtime_error("[TorchCommMSCCLPP] Unsupported tensor dtype: " + std::string(at::toString(dtype)) +
                               ". Supported: float32, float16, bfloat16, int32, uint32.");
  }
}

// Maps TorchComms ReduceOp to MSCCL++ ReduceOp.
// Currently only SUM and MIN are supported by MSCCL++ native kernels.
// When MSCCL++ adds more reduction ops, extend this mapping.
mscclpp::ReduceOp TorchCommMSCCLPP::torchReduceOpToMscclpp(const ReduceOp& op, const std::string& collective_name) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
      return mscclpp::SUM;
    case ReduceOp::RedOpType::MIN:
      return mscclpp::MIN;
    default:
      throw std::runtime_error("[TorchCommMSCCLPP] " + collective_name +
                               " does not support the requested reduction op (type=" +
                               std::to_string(static_cast<int>(op.type())) + "). Supported: SUM, MIN.");
  }
}

// Async ops use the dedicated internal stream so the call returns immediately
// without blocking work on the caller's stream. Sync ops use the caller's
// current PyTorch CUDA stream so the executor launch is ordered inline with
// any preceding work on that stream.
cudaStream_t TorchCommMSCCLPP::getOperationStream(bool async_op) const {
  if (async_op) {
    return internal_stream_;
  }
  return at::cuda::getCurrentCUDAStream(device_.index()).stream();
}

void TorchCommMSCCLPP::checkInitialized() const {
  if (!initialized_) {
    throw std::runtime_error("[TorchCommMSCCLPP] Communicator not initialized. Call init() first.");
  }
}

// --- Lifecycle ---

TorchCommMSCCLPP::TorchCommMSCCLPP() = default;

TorchCommMSCCLPP::~TorchCommMSCCLPP() {
  if (initialized_) {
    // Best-effort cleanup if user forgot finalize()
    try {
      finalize();
    } catch (...) {
    }
  }
}

void TorchCommMSCCLPP::init(at::Device device, const std::string& name, const CommOptions& options) {
  if (initialized_) {
    throw std::runtime_error("[TorchCommMSCCLPP] Already initialized. Call finalize() first.");
  }

  device_ = device;
  name_ = name;
  options_ = options;

  // 1. Bootstrap: discovers rank/size and creates the Communicator
  auto bootstrap = std::make_unique<TorchCommMSCCLPPBootstrap>(options.store, device, options.timeout);
  rank_ = bootstrap->getRank();
  size_ = bootstrap->getSize();
  comm_ = bootstrap->createCommunicator(name, options);

  // 2. Select GPU device
  MSCCLPP_CUDATHROW(cudaSetDevice(device_.index()));

  // 3. Cache nRanksPerNode
  nRanksPerNode_ = comm_->bootstrap()->getNranksPerNode();

  // 4. Create dedicated internal stream for async operations
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&internal_stream_, cudaStreamNonBlocking));

  // 5. Allocate scratch buffer using GpuBuffer (cuMemMap on NVLS-capable GPUs).
  // GpuBuffer registers POSIX file descriptors in the unix socket server,
  // which is required for cross-rank IPC sharing of the scratch buffer.
  // Plain cudaMalloc does NOT register fds, causing "Requested fd not found" crashes.
  scratchBuffer_ = mscclpp::GpuBuffer<char>(kScratchBufferSize).memory();

  // 6. Create Executor with the scratch buffer (same as NCCL extension).
  // The Executor uses this as its defaultScratchBuffer for DSL plans.
  executor_ = std::make_shared<mscclpp::Executor>(comm_, scratchBuffer_);

  // 7. Get flag buffer and keep it alive for the lifetime of the communicator.
  auto [flagBuf, flagSize] = mscclpp::getFlagBuffer();
  flagBuffer_ = flagBuf;
  flagBufferSize_ = flagSize;

  // 8. Build AlgorithmCollection with default native + DSL algorithms.
  //
  // TODO: The algorithm selector logic below is duplicated from
  // the NCCL extension (src/ext/nccl/nccl.cc). It should be moved into
  // AlgorithmCollectionBuilder::buildDefaultAlgorithms() so that all consumers
  // (NCCL ext, torchcomms, Python API) get a default selector automatically
  // without having to wire one up themselves.
  //
  // We use the same algorithm selector as the NCCL/RCCL compatibility layer —
  // it has proper topology-aware selection logic considering message size, NVLS
  // support, compute capability, symmetric memory, and CUDA graph mode.
  auto builder = mscclpp::collective::AlgorithmCollectionBuilder::getInstance();

  // Detect hardware capabilities for algorithm selection
  static const bool isNvlsSupported = mscclpp::isNvlsSupported();
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int major = 0, minor = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cudaDevice));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cudaDevice));
  static const std::pair<int, int> computeCapability = {major, minor};

  auto algoSelector =
      [](const std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<mscclpp::Algorithm>>>&
             algoMapByCollective,
         const mscclpp::CollectiveRequest& request) -> std::shared_ptr<mscclpp::Algorithm> {
    auto collectiveIt = algoMapByCollective.find(request.collective);
    if (collectiveIt == algoMapByCollective.end()) {
      return nullptr;
    }

    const bool isCuMemMapAllocated = mscclpp::isCuMemMapAllocated(const_cast<void*>(request.inputBuffer)) &&
                                     mscclpp::isCuMemMapAllocated(request.outputBuffer);

    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(request.stream, &captureStatus);
    const bool inCaptureMode = (captureStatus == cudaStreamCaptureStatusActive);

    mscclpp::nccl::AlgorithmSelectorConfig config{
        .symmetricMemory = false,
        // nvlsSupported reflects hardware capability only (same as NCCL ext).
        // Non-zero-copy NVLS algorithms (warp_pipeline, block_pipeline) work
        // with regular cudaMalloc tensors — they allocate their own NVLS
        // multicast memory internally. Only zero-copy variants need cuMemMap
        // input/output buffers, which is gated by useNvlsWithZeroCopy in the
        // selector (requires both symmetricMemory AND isCuMemMapAllocated).
        .nvlsSupported = isNvlsSupported,
        .isCuMemMapAllocated = isCuMemMapAllocated,
        .inCaptureMode = inCaptureMode,
        .computeCapability = computeCapability,
        .ncclDlopenSharedLib = false,
    };

    const auto& algoMap = collectiveIt->second;

    // Multi-node: native algorithm selector returns nullptr (not yet implemented).
    // DSL plans may handle specific multi-node configurations (e.g., 2-node 8-GPU allreduce).
    if (request.nRanksPerNode != request.worldSize) {
      return mscclpp::nccl::selectMultiNodeAlgorithm(algoMap, request, config);
    }

    if (request.collective == "allgather") {
      return mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config);
    }
    if (request.collective == "allreduce") {
      return mscclpp::nccl::selectSingleNodeAllreduce(algoMap, request, config);
    }

    // For other collectives (reducescatter, alltoall), try DSL plans
    for (const auto& [name, algo] : algoMap) {
      if (algo->type() == mscclpp::AlgorithmType::DSL) {
        auto dslAlgo = std::dynamic_pointer_cast<mscclpp::DslAlgorithm>(algo);
        if (dslAlgo && mscclpp::nccl::matchExecutionPlan(dslAlgo, request)) {
          return algo;
        }
      }
    }
    return nullptr;
  };

  builder->setFallbackAlgorithmSelector(algoSelector);
  algorithmCollection_ =
      builder->buildDefaultAlgorithms(reinterpret_cast<uintptr_t>(scratchBuffer_.get()), kScratchBufferSize,
                                      reinterpret_cast<uintptr_t>(flagBuffer_.get()), flagBufferSize_, rank_);

  // 9. Create GPU event pool
  event_pool_ = std::make_shared<MscclppGpuEventPool>(256);

  initialized_ = true;
}

void TorchCommMSCCLPP::finalize() {
  if (!initialized_) {
    return;
  }

  // Drain our own streams while the communicator (and NVLink memory) is alive.
  // After work.wait() (which is GPU-side only), this rank's collective kernel
  // is done. But ring-algorithm collectives may finish on different ranks at
  // slightly different times — one rank can complete while another's kernel is
  // still polling NVLink memory.
  //
  // Teardown sequence:
  //   1. Sync our own streams (fast — work is already done per wait())
  //   2. bootstrap barrier: CPU rendezvous ensures ALL ranks have drained
  //      their GPU work before ANY rank destroys its communicator
  //   3. CPU-side teardown in reverse init order
  if (internal_stream_) {
    cudaStreamSynchronize(internal_stream_);
  }
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(device_.index()).stream());

  // All ranks rendezvous here. Once every rank returns from this barrier,
  // no NVLink-polling kernel is running anywhere, so comm_.reset() is safe.
  comm_->bootstrap()->barrier();

  // Teardown in reverse init order
  executor_.reset();
  event_pool_.reset();

  if (internal_stream_) {
    cudaStreamDestroy(internal_stream_);
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

// --- Collective execution (unified path) ---
//
// All supported collectives funnel through executeCollective(). This method:
//   1. Builds a CollectiveRequest describing the operation (world size, message
//      size, dtype, buffer pointers, etc.)
//   2. Asks AlgorithmCollection to select the best algorithm — this considers
//      message size, topology (world size, nRanksPerNode), and buffer mode
//      (in-place vs out-of-place). The collection contains both native C++/CUDA
//      algorithms (fastest, compiled kernels) and DSL algorithms (flexible,
//      JSON execution plans). The backend doesn't need to know which type runs.
//   3. Creates a TorchWorkMSCCLPP handle with GPU start/end events
//   4. Calls algo->execute() which either launches a native kernel directly
//      or interprets a DSL plan through the Executor
//   5. Returns the work handle — caller uses work->wait() for GPU-side sync

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::executeCollective(const std::string& collective, const void* sendbuf,
                                                                  void* recvbuf, size_t sendBytes, size_t recvBytes,
                                                                  mscclpp::DataType dtype, mscclpp::ReduceOp reduceOp,
                                                                  bool async_op, std::chrono::milliseconds timeout) {
  std::unordered_map<std::string, std::vector<uint64_t>> hints;
  mscclpp::CollectiveRequest request{
      size_, nRanksPerNode_, rank_, sendbuf, recvbuf, sendBytes, getOperationStream(async_op), collective, dtype, hints,
  };

  auto algo = algorithmCollection_.selectAlgorithm(request);
  if (!algo) {
    throw std::runtime_error("[TorchCommMSCCLPP] No algorithm registered for collective '" + collective +
                             "' with message size " + std::to_string(sendBytes));
  }

  auto stream = getOperationStream(async_op);

  auto work = c10::make_intrusive<TorchWorkMSCCLPP>(stream, device_.index(), timeout, event_pool_);
  work->recordStart();

  // Always pass executor_ — native algorithms ignore it, DSL algorithms need
  // it to interpret JSON execution plans.
  algo->execute(comm_, sendbuf, recvbuf, sendBytes, recvBytes, dtype, reduceOp, stream, executor_);

  work->recordEnd();
  return work;
}

// --- Supported collectives ---
//
// Each supported collective: validates inputs → ensures contiguous → calls
// executeCollective() with the MSCCL++ collective name and buffer pointers.
// MSCCL++ collective names: "allreduce", "allgather", "reducescatter", etc.

// AllReduce: in-place SUM reduction across all ranks.
// Input and output are the same buffer (in-place operation).
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_reduce(at::Tensor& tensor, const ReduceOp& op, bool async_op,
                                                           const AllReduceOptions& options) {
  checkInitialized();
  auto mscclppOp = torchReduceOpToMscclpp(op, "all_reduce");
  tensor = tensor.contiguous();

  return executeCollective("allreduce", tensor.data_ptr(), tensor.data_ptr(), tensor.nbytes(), tensor.nbytes(),
                           torchDtypeToMscclpp(tensor.scalar_type()), mscclppOp, async_op, options.timeout);
}

// AllGatherSingle: each rank contributes input -> output has all ranks' data concatenated.
// The sendbuf is the input chunk, recvbuf is the full output buffer.
// The MSCCL++ allgather algorithm handles placing each rank's chunk internally.
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_single(at::Tensor& output, const at::Tensor& input,
                                                                  bool async_op,
                                                                  const AllGatherSingleOptions& options) {
  checkInitialized();
  auto input_contig = input.contiguous();
  output = output.contiguous();

  const size_t chunk_bytes = static_cast<size_t>(input_contig.nbytes());

  return executeCollective("allgather", input_contig.data_ptr(), output.data_ptr(), chunk_bytes,
                           static_cast<size_t>(output.nbytes()), torchDtypeToMscclpp(input_contig.scalar_type()),
                           mscclpp::NOP, async_op, options.timeout);
}

// ReduceScatterSingle: SUM-reduce input across all ranks, then scatter the
// result so each rank gets its chunk. Input is the full buffer, output is
// this rank's reduced chunk.
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_single(at::Tensor& output, const at::Tensor& input,
                                                                      const ReduceOp& op, bool async_op,
                                                                      const ReduceScatterSingleOptions& options) {
  checkInitialized();
  auto mscclppOp = torchReduceOpToMscclpp(op, "reduce_scatter_single");
  auto input_contig = input.contiguous();
  output = output.contiguous();

  return executeCollective("reducescatter", input_contig.data_ptr(), output.data_ptr(),
                           static_cast<size_t>(input_contig.nbytes()), static_cast<size_t>(output.nbytes()),
                           torchDtypeToMscclpp(input_contig.scalar_type()), mscclppOp, async_op, options.timeout);
}

// AllToAllSingle: each rank sends its i-th chunk to rank i and receives
// rank i's chunk into its own i-th output slot. Full permutation.
c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_single(at::Tensor& output, const at::Tensor& input,
                                                                  bool async_op, const AllToAllSingleOptions& options) {
  checkInitialized();
  auto input_contig = input.contiguous();
  output = output.contiguous();

  return executeCollective("alltoall", input_contig.data_ptr(), output.data_ptr(),
                           static_cast<size_t>(input_contig.nbytes()), static_cast<size_t>(output.nbytes()),
                           torchDtypeToMscclpp(input_contig.scalar_type()), mscclpp::NOP, async_op, options.timeout);
}

// --- Unsupported operations ---
//
// MSCCL++ focuses on high-performance allreduce/allgather/reducescatter/alltoall.
// Operations below are not supported — each throws with an explicit message
// suggesting the caller use a separate NCCL (NVIDIA) or RCCL (AMD) communicator.
// This is the recommended pattern for mixed-backend training: use MSCCL++ for
// the hot collectives (gradient allreduce, etc.) and NCCL/RCCL for the rest.

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::send(const at::Tensor&, int, bool, const SendOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] send() is not supported. "
      "Use a separate NCCL/RCCL communicator for point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::recv(at::Tensor&, int, bool, const RecvOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] recv() is not supported. "
      "Use a separate NCCL/RCCL communicator for point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::batch_op_issue(const std::vector<BatchSendRecv::P2POp>&, bool,
                                                               const BatchP2POptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] batch_op_issue() is not supported. "
      "Use a separate NCCL/RCCL communicator for batched point-to-point.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::broadcast(at::Tensor&, int, bool, const BroadcastOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] broadcast() is not supported. "
      "Use a separate NCCL/RCCL communicator for broadcast.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce(const at::Tensor&, int, const ReduceOp&, bool,
                                                       const ReduceOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce() is not supported. "
      "Use a separate NCCL/RCCL communicator for reduce.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather(const std::vector<at::Tensor>&, const at::Tensor&, bool,
                                                           const AllGatherOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather() (tensor-list variant) is not supported. "
      "Use all_gather_single() instead, or a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_gather_v(const std::vector<at::Tensor>&, const at::Tensor&, bool,
                                                             const AllGatherOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_gather_v() is not supported. "
      "Use a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter(at::Tensor&, const std::vector<at::Tensor>&,
                                                               const ReduceOp&, bool, const ReduceScatterOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter() (tensor-list variant) is not supported. "
      "Use reduce_scatter_single() instead, or a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::reduce_scatter_v(at::Tensor&, const std::vector<at::Tensor>&,
                                                                 const ReduceOp&, bool, const ReduceScatterOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] reduce_scatter_v() is not supported. "
      "Use a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all_v_single(at::Tensor&, const at::Tensor&,
                                                                    const std::vector<uint64_t>&,
                                                                    const std::vector<uint64_t>&, bool,
                                                                    const AllToAllvSingleOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all_v_single() is not supported. "
      "Use a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::all_to_all(const std::vector<at::Tensor>&,
                                                           const std::vector<at::Tensor>&, bool,
                                                           const AllToAllOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] all_to_all() (tensor-list variant) is not supported. "
      "Use all_to_all_single() instead, or a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::barrier(bool, const BarrierOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] barrier() is not supported. "
      "Use a separate NCCL/RCCL communicator for barrier.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::scatter(at::Tensor&, const std::vector<at::Tensor>&, int, bool,
                                                        const ScatterOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] scatter() is not supported. "
      "Use a separate NCCL/RCCL communicator.");
}

c10::intrusive_ptr<TorchWork> TorchCommMSCCLPP::gather(const std::vector<at::Tensor>&, const at::Tensor&, int, bool,
                                                       const GatherOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] gather() is not supported. "
      "Use a separate NCCL/RCCL communicator.");
}

std::shared_ptr<TorchCommBackend> TorchCommMSCCLPP::split(const std::vector<int>&, const std::string&,
                                                          const CommOptions&) {
  throw std::runtime_error(
      "[TorchCommMSCCLPP] split() is not supported. "
      "Use a separate NCCL/RCCL communicator that supports sub-communicators.");
}

// --- Factory registration ---
//
// Registers "mscclpp" as a backend name with TorchCommFactory.
//
// From Python:  comm = torchcomms.new_comm("mscclpp", device, name="grad_sync")
// From C++:     auto backend = TorchCommFactory::get().create_backend("mscclpp", device, name);
//
// The factory calls this lambda to instantiate a TorchCommMSCCLPP, then the
// caller invokes init() which triggers the full bootstrap + setup flow.

namespace {
class MSCCLPPRegistration {
 public:
  MSCCLPPRegistration() {
    TorchCommFactory::get().register_backend("mscclpp", []() { return std::make_shared<TorchCommMSCCLPP>(); });
  }
};
static const MSCCLPPRegistration registration{};
}  // namespace

}  // namespace torch::comms
