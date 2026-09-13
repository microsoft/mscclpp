// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nvml.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "megamoe.hpp"

namespace mscclpp::megamoe {
namespace {

size_t product(std::initializer_list<size_t> factors) {
  size_t result = 1;
  for (auto factor : factors) {
    if (factor && result > std::numeric_limits<size_t>::max() / factor)
      throw std::invalid_argument("MegaMoE buffer size overflows size_t");
    result *= factor;
  }
  return result;
}

struct RankInfo {
  std::array<int, 9> config;
  float gateUpClamp;
  char gpuUuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
  unsigned char fabricUuid[NVML_GPU_FABRIC_UUID_LEN]{};
  unsigned int cliqueId = 0;
  bool fabricReady = false;
};

RankInfo rankInfo(const NativeConfig& c, int device) {
  RankInfo info{};
  info.config = {c.worldSize, c.maxTokens,       c.hidden, c.intermediate, c.numExperts, c.topK,
                 c.smMargin,  int(c.weightE5M2), 1};
  info.gateUpClamp = c.gateUpClamp;
  if (c.worldSize == 1) return info;
  if (nvmlInit_v2() != NVML_SUCCESS)
    throw std::runtime_error("MegaMoE requires NVML to verify that all peers share an NVLink fabric");
  struct NvmlGuard {
    ~NvmlGuard() { (void)nvmlShutdown(); }
  } guard;
  char pciBusId[32]{};
  MSCCLPP_CUDATHROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));
  nvmlDevice_t handle;
  if (nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &handle) != NVML_SUCCESS ||
      nvmlDeviceGetUUID(handle, info.gpuUuid, sizeof(info.gpuUuid)) != NVML_SUCCESS)
    throw std::runtime_error("MegaMoE could not identify the current GPU through NVML");
#if defined(nvmlGpuFabricInfo_v2)
  nvmlGpuFabricInfoV_t fabric{};
  fabric.version = nvmlGpuFabricInfo_v2;
  auto result = nvmlDeviceGetGpuFabricInfoV(handle, &fabric);
#else
  nvmlGpuFabricInfo_t fabric{};
  auto result = nvmlDeviceGetGpuFabricInfo(handle, &fabric);
#endif
  if (result == NVML_SUCCESS && fabric.state == NVML_GPU_FABRIC_STATE_COMPLETED && fabric.status == NVML_SUCCESS) {
    std::memcpy(info.fabricUuid, fabric.clusterUuid, sizeof(info.fabricUuid));
    info.cliqueId = fabric.cliqueId;
    info.fabricReady = std::any_of(std::begin(info.fabricUuid), std::end(info.fabricUuid),
                                   [](unsigned char value) { return value != 0; });
  }
  return info;
}

void validatePeers(const std::shared_ptr<Communicator>& comm, const NativeConfig& c, int device) {
  std::vector<RankInfo> peers(c.worldSize);
  peers[c.rank] = rankInfo(c, device);
  comm->bootstrap()->allGather(peers.data(), sizeof(RankInfo));
  const auto& local = peers[c.rank];
  for (int rank = 0; rank < c.worldSize; ++rank) {
    const auto& peer = peers[rank];
    if (peer.config != local.config || peer.gateUpClamp != local.gateUpClamp)
      throw std::invalid_argument("MegaMoE configurations must match on every rank (except rank)");
    if (c.worldSize > 1 && (!peer.fabricReady || !local.fabricReady || peer.cliqueId != local.cliqueId ||
                            std::memcmp(peer.fabricUuid, local.fabricUuid, sizeof(local.fabricUuid))))
      throw std::invalid_argument(
          "MegaMoE requires all peers in one active NVLink fabric; PCIe/IB peers are unsupported");
    for (int previous = 0; previous < rank; ++previous)
      if (!std::strcmp(peer.gpuUuid, peers[previous].gpuUuid))
        throw std::invalid_argument("MegaMoE requires exactly one rank per GPU");
  }
}

bool overlaps(const void* a, size_t aBytes, const void* b, size_t bBytes) {
  auto x = reinterpret_cast<uintptr_t>(a);
  auto y = reinterpret_cast<uintptr_t>(b);
  return aBytes && bBytes && (x <= y ? y - x < aBytes : x - y < bBytes);
}

void stage(void* destination, const void* source, size_t bytes, cudaStream_t stream) {
  if (!bytes || source == destination) return;
  if (!source || overlaps(source, bytes, destination, bytes))
    throw std::invalid_argument("MegaMoE input must be non-null and not partially overlap its staging buffer");
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

struct MegaMoeContext::Impl {
  NativeConfig config;
  int device = -1;
  SymmetricLayout layout;
  size_t workspaceBytes = 0;
  std::shared_ptr<Communicator> communicator;
  std::unique_ptr<GpuBuffer<char>> symmetric;
  std::shared_ptr<char> workspace;
  std::shared_ptr<uint64_t> peerBases;
  std::shared_ptr<uint32_t> startSignal;
  cudaEvent_t startResetEvent = nullptr;
  bool startRecorded = false;
  std::array<std::shared_ptr<uint8_t>, 4> weightBuffers;
  RegisteredMemory localMemory;
  std::vector<RegisteredMemory> peerMemories;
  std::vector<Connection> connections;
  std::shared_ptr<KernelPlan> plan;

  ~Impl() {
    if (device < 0) return;
    // Synchronize before destroying plans/imports/allocations; member destruction
    // alone would unmap peer memory before cudaFree waits for pending work.
    int original = -1;
    (void)cudaGetDevice(&original);
    (void)cudaSetDevice(device);
    (void)cudaDeviceSynchronize();
    if (startResetEvent) (void)cudaEventDestroy(startResetEvent);
    plan.reset();
    connections.clear();
    peerMemories.clear();
    localMemory = RegisteredMemory{};
    weightBuffers = {};
    peerBases.reset();
    workspace.reset();
    symmetric.reset();
    if (original >= 0) (void)cudaSetDevice(original);
  }
};

MegaMoeContext::MegaMoeContext(std::shared_ptr<Communicator> comm, const NativeConfig& c, const PackedWeights& weights,
                               cudaStream_t stream, int tag)
    : impl_(std::make_unique<Impl>()) {
  validateNativeConfig(c);
  if (!comm || comm->bootstrap()->getRank() != c.rank || comm->bootstrap()->getNranks() != c.worldSize)
    throw std::invalid_argument("MegaMoE rank/worldSize must match its communicator");
  if (tag < 0 || tag == std::numeric_limits<int>::max())
    throw std::invalid_argument("MegaMoE bootstrap tag must be in [0, INT_MAX-1)");
  if (!weights.fc1 || !weights.fc1Scale || !weights.fc2 || !weights.fc2Scale)
    throw std::invalid_argument("MegaMoE requires four non-null canonical weight buffers");
  cudaStreamCaptureStatus capture;
  MSCCLPP_CUDATHROW(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("Construct MegaMoE outside CUDA Graph capture");
  auto& p = *impl_;
  MSCCLPP_CUDATHROW(cudaGetDevice(&p.device));
  cudaDeviceProp properties{};
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&properties, p.device));
  if (properties.major != 10 || properties.minor != 0)
    throw std::invalid_argument("Native MegaMoE requires an SM100 GPU (for example GB200)");
  if (c.smMargin > properties.multiProcessorCount - 2)
    throw std::invalid_argument("MegaMoE smMargin must leave at least two SMs");
  validatePeers(comm, c, p.device);
  p.config = c;
  p.communicator = std::move(comm);
  p.layout = getSymmetricLayout(c);
  p.workspaceBytes = getPrivateWorkspaceBytes(c);
  p.symmetric = std::make_unique<GpuBuffer<char>>(p.layout.bytes);
  p.workspace = detail::gpuCallocShared<char>(p.workspaceBytes);
  p.peerBases = detail::gpuCallocShared<uint64_t>(c.worldSize);
  p.startSignal = detail::gpuCallocShared<uint32_t>(1);
  MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&p.startResetEvent, cudaEventDisableTiming));
  p.localMemory = p.communicator->registerMemory(p.symmetric->data(), p.layout.bytes, Transport::CudaIpc);
  p.peerMemories.resize(c.worldSize);
  p.peerMemories[c.rank] = p.localMemory;
  std::vector<std::shared_future<RegisteredMemory>> memoryFutures(c.worldSize);
  std::vector<std::shared_future<Connection>> connectionFutures(c.worldSize);
  for (int rank = 0; rank < c.worldSize; ++rank) {
    if (rank == c.rank) continue;
    connectionFutures[rank] = p.communicator->connect(Transport::CudaIpc, rank, tag);
    p.communicator->sendMemory(p.localMemory, rank, tag + 1);
    memoryFutures[rank] = p.communicator->recvMemory(rank, tag + 1);
  }
  std::vector<uint64_t> peerPointers(c.worldSize);
  for (int rank = 0; rank < c.worldSize; ++rank) {
    if (rank != c.rank) {
      p.connections.push_back(connectionFutures[rank].get());
      p.peerMemories[rank] = memoryFutures[rank].get();
    }
    if (!p.peerMemories[rank].data() || p.peerMemories[rank].size() != p.layout.bytes)
      throw std::runtime_error("MegaMoE could not map a matching CudaIpc peer workspace");
    peerPointers[rank] = reinterpret_cast<uint64_t>(p.peerMemories[rank].data());
  }
  size_t localExperts = c.numExperts / c.worldSize;
  const std::array<size_t, 4> sizes{product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden)}),
                                    product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden / 32)}),
                                    product({localExperts, size_t(c.hidden), size_t(c.intermediate)}),
                                    product({localExperts, size_t(c.hidden), size_t(c.intermediate / 32)})};
  for (size_t i = 0; i < sizes.size(); ++i) p.weightBuffers[i] = detail::gpuCallocShared<uint8_t>(sizes[i]);
  PackedWeights packed{p.weightBuffers[0].get(), p.weightBuffers[1].get(), p.weightBuffers[2].get(),
                       p.weightBuffers[3].get()};
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(p.peerBases.get(), peerPointers.data(), c.worldSize * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, stream));
  packNativeWeights(c, weights, packed, stream);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  p.plan = createKernelPlan(c, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), packed);
  p.communicator->bootstrap()->barrier();
}

MegaMoeContext::~MegaMoeContext() = default;
const NativeConfig& MegaMoeContext::config() const { return impl_->config; }
int MegaMoeContext::device() const { return impl_->device; }
void* MegaMoeContext::input() const { return impl_->symmetric->data() + impl_->layout.input; }
void* MegaMoeContext::topkIds() const { return impl_->symmetric->data() + impl_->layout.topkIds; }
void* MegaMoeContext::topkWeights() const { return impl_->symmetric->data() + impl_->layout.topkWeights; }
int MegaMoeContext::ctaCount() const { return kernelPlanCtaCount(*impl_->plan); }
size_t MegaMoeContext::sharedBytes() const { return kernelPlanSharedBytes(*impl_->plan); }
size_t MegaMoeContext::symmetricBytes() const { return impl_->layout.bytes; }
size_t MegaMoeContext::privateBytes() const { return impl_->workspaceBytes; }

size_t MegaMoeContext::validateForward(const void* x, void* output, int tokens) const {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE forward must use the context's CUDA device");
  if (tokens < 0 || tokens > config().maxTokens)
    throw std::invalid_argument("MegaMoE numTokens exceeds the configured capacity");
  size_t inputBytes = size_t(tokens) * config().hidden * 2;
  if ((tokens && (!x || !output)) || overlaps(output, inputBytes, impl_->symmetric->data(), impl_->layout.bytes))
    throw std::invalid_argument("MegaMoE requires non-null arrays and output disjoint from the registered workspace");
  return inputBytes;
}

void MegaMoeContext::forward(const void* x, const int32_t* ids, const float* scores, void* output, int tokens,
                             cudaStream_t stream, bool signalStart) {
  const size_t inputBytes = validateForward(x, output, tokens);
  const size_t routingBytes = size_t(tokens) * config().topK * 4;
  if (tokens && (!ids || !scores)) throw std::invalid_argument("MegaMoE routing arrays must be non-null");
  stage(input(), x, inputBytes, stream);
  stage(topkIds(), ids, routingBytes, stream);
  stage(topkWeights(), scores, routingBytes, stream);
  impl_->startRecorded = false;
  if (signalStart) {
    MSCCLPP_CUDATHROW(cudaMemsetAsync(impl_->startSignal.get(), 0, sizeof(uint32_t), stream));
    MSCCLPP_CUDATHROW(cudaEventRecord(impl_->startResetEvent, stream));
  }
  launchNativeMegaMoe(impl_->plan, tokens, output, stream, signalStart ? impl_->startSignal.get() : nullptr);
  impl_->startRecorded = signalStart;
}

void MegaMoeContext::forwardShared(const void* x, void* output, int tokens, cudaStream_t stream) {
  if (config().worldSize != 1 || config().numExperts != 1 || config().topK != 1)
    throw std::invalid_argument("Shared forward requires world_size=1, num_experts=1, top_k=1");
  const size_t inputBytes = validateForward(x, output, tokens);
  impl_->startRecorded = false;
  stage(input(), x, inputBytes, stream);
  launchNativeSharedExpert(impl_->plan, tokens, output, stream);
}

void MegaMoeContext::waitUntilStarted(cudaStream_t stream) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE start wait must use the context's CUDA device");
  if (!impl_->startRecorded)
    throw std::invalid_argument("MegaMoE waitUntilStarted requires a preceding signalStart forward");
  MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, impl_->startResetEvent, 0));
  MSCCLPP_CUTHROW(
      cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(impl_->startSignal.get()), 1, CU_STREAM_WAIT_VALUE_EQ));
}

}  // namespace mscclpp::megamoe
