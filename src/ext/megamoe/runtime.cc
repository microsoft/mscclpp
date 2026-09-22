// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe.hpp"
#include "megamoe_jit.hpp"

namespace mscclpp::megamoe {
namespace {

constexpr size_t ErrorCapacity = 1024;

struct RankStatus {
  int code = 0;
  char message[ErrorCapacity]{};
};

template <class Function>
void collectiveCheck(const std::shared_ptr<Communicator>& comm, const char* phase, Function&& function) {
  const int rank = comm->bootstrap()->getRank();
  const int ranks = comm->bootstrap()->getNranks();
  std::vector<RankStatus> statuses(ranks);
  auto& local = statuses[rank];
  auto report = [&](int code, const char* message) {
    local.code = code;
    std::strncpy(local.message, message, sizeof(local.message) - 1);
  };
  try {
    function();
  } catch (const std::invalid_argument& e) {
    report(1, e.what());
  } catch (const std::exception& e) {
    report(2, e.what());
  } catch (...) {
    report(2, "Unknown C++ exception");
  }
  comm->bootstrap()->allGather(statuses.data(), sizeof(RankStatus));
  for (int peer = 0; peer < ranks; ++peer) {
    const auto& status = statuses[peer];
    if (!status.code) continue;
    std::string message =
        "MegaMoE " + std::string(phase) + " failed on rank " + std::to_string(peer) + ": " + status.message;
    if (status.code == 1) throw std::invalid_argument(message);
    throw std::runtime_error(message);
  }
}

bool validKernelId(const std::string& id) {
  return id.size() == MSCCLPP_MEGAMOE_JIT_ID_CAPACITY - 1 &&
         std::all_of(id.begin(), id.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

void checkJit(int status, const char* error, const char* operation) {
  if (!status) return;
  std::string message = std::string("MegaMoE JIT ") + operation + ": " +
                        (error[0] ? error : "module returned an error without a diagnostic");
  if (status == 1) throw std::invalid_argument(message);
  throw std::runtime_error(message);
}

MegaMoeJitConfigV1 jitConfig(const NativeConfig& c) {
  return MegaMoeJitConfigV1{c.rank,       c.worldSize, c.maxTokens, c.hidden,          c.intermediate,
                            c.numExperts, c.topK,      c.smMargin,  int(c.weightE5M2), c.gateUpClamp};
}

MegaMoeJitWeightsV1 jitWeights(const PackedWeights& weights) {
  return MegaMoeJitWeightsV1{weights.fc1, weights.fc1Scale, weights.fc2, weights.fc2Scale};
}

struct JitModule {
  void* handle = nullptr;
  const MegaMoeJitApiV1* api = nullptr;

  ~JitModule() {
    if (handle) (void)dlclose(handle);
  }

  void load(const std::string& path, const std::string& id) {
    if (!validKernelId(id))
      throw std::invalid_argument("MegaMoE kernel_id must be a 64-character lowercase hex digest");
    if (path.empty() || path.front() != '/')
      throw std::invalid_argument("MegaMoE kernel_path must be an explicit absolute path");
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error("Could not load MegaMoE JIT module " + path + ": " + dlerror());
    (void)dlerror();
    auto entry = reinterpret_cast<MegaMoeJitGetApiV1>(dlsym(handle, MSCCLPP_MEGAMOE_JIT_ENTRYPOINT));
    const char* error = dlerror();
    if (error || !entry)
      throw std::runtime_error(std::string("MegaMoE JIT entrypoint is missing: ") + (error ? error : "null symbol"));
    const auto* api = entry();
    if (!api || api->abiVersion != MSCCLPP_MEGAMOE_JIT_ABI_VERSION || api->structBytes != sizeof(MegaMoeJitApiV1))
      throw std::invalid_argument("MegaMoE JIT module ABI version or function-table size mismatch");
    if (api->configBytes != sizeof(MegaMoeJitConfigV1) || api->weightsBytes != sizeof(MegaMoeJitWeightsV1) ||
        api->layoutBytes != sizeof(MegaMoeJitLayoutV1) || (api->tileM != 128 && api->tileM != 256) ||
        (api->tileM == 128 && api->tileK == 32) || (api->tileK != 32 && api->tileK != 64 && api->tileK != 128) ||
        api->clusterSize != detail::ClusterM || api->accumulatorStages != 2 || api->architecture != 1000)
      throw std::invalid_argument("MegaMoE JIT module metadata does not match the native ABI");
    if (std::memcmp(api->kernelId, id.c_str(), MSCCLPP_MEGAMOE_JIT_ID_CAPACITY))
      throw std::invalid_argument("MegaMoE JIT module kernel_id does not match the requested specialization");
    const bool supported = (api->tileN == 32 && api->loadStages == 8 && api->transformStages == 7) ||
                           (api->tileN == 32 && api->loadStages == 6 && api->transformStages == 7) ||
                           (api->tileN == 64 && api->loadStages == 6 && api->transformStages == 6) ||
                           (api->tileN == 128 && api->loadStages == 4 && api->transformStages == 4);
    if (!supported) throw std::invalid_argument("MegaMoE JIT module declares an unsupported specialization");
    if (!api->preflight || !api->packWeights || !api->createPlan || !api->destroyPlan || !api->launch)
      throw std::invalid_argument("MegaMoE JIT module has an incomplete function table");
    this->api = api;
  }
};

struct JitPlan {
  std::shared_ptr<JitModule> module;
  void* handle = nullptr;

  explicit JitPlan(std::shared_ptr<JitModule> owner) : module(std::move(owner)) {}
  ~JitPlan() {
    if (handle) module->api->destroyPlan(handle);
  }
};

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
  int tag = 0;
  std::array<int, 5> specialization;
  char kernelId[MSCCLPP_MEGAMOE_JIT_ID_CAPACITY]{};
  size_t symmetricBytes = 0;
  size_t privateBytes = 0;
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

void validatePeers(const std::shared_ptr<Communicator>& comm, const NativeConfig& c, const RankInfo& info) {
  std::vector<RankInfo> peers(c.worldSize);
  peers[c.rank] = info;
  comm->bootstrap()->allGather(peers.data(), sizeof(RankInfo));
  const auto& local = peers[c.rank];
  for (int rank = 0; rank < c.worldSize; ++rank) {
    const auto& peer = peers[rank];
    if (peer.config != local.config || peer.gateUpClamp != local.gateUpClamp || peer.tag != local.tag)
      throw std::invalid_argument("MegaMoE configurations and bootstrap tags must match on every rank (except rank)");
    if (peer.specialization != local.specialization ||
        std::memcmp(peer.kernelId, local.kernelId, sizeof(local.kernelId)) ||
        peer.symmetricBytes != local.symmetricBytes || peer.privateBytes != local.privateBytes)
      throw std::invalid_argument(
          "MegaMoE kernel IDs, specialization metadata, and workspace sizes must match on every rank");
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
  std::string kernelId = "builtin";
  KernelResources resources;
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
  std::shared_ptr<JitModule> module;
  std::unique_ptr<JitPlan> jitPlan;

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
    jitPlan.reset();
    module.reset();
    connections.clear();
    peerMemories.clear();
    localMemory = RegisteredMemory{};
    weightBuffers = {};
    startSignal.reset();
    peerBases.reset();
    workspace.reset();
    symmetric.reset();
    if (original >= 0) (void)cudaSetDevice(original);
  }
};

MegaMoeContext::MegaMoeContext(std::shared_ptr<Communicator> comm, const NativeConfig& c, const PackedWeights& weights,
                               cudaStream_t stream, int tag, const std::string& kernelPath, const std::string& kernelId)
    : impl_(std::make_unique<Impl>()) {
  if (!comm || !comm->bootstrap()) throw std::invalid_argument("MegaMoE requires a communicator with a bootstrap");
  auto& p = *impl_;
  RankInfo info{};
  collectiveCheck(comm, "preflight", [&] {
    validateNativeConfig(c);
    if (comm->bootstrap()->getRank() != c.rank || comm->bootstrap()->getNranks() != c.worldSize)
      throw std::invalid_argument("MegaMoE rank/worldSize must match its communicator");
    if (tag < 0 || tag == std::numeric_limits<int>::max())
      throw std::invalid_argument("MegaMoE bootstrap tag must be in [0, INT_MAX-1)");
    if (!weights.fc1 || !weights.fc1Scale || !weights.fc2 || !weights.fc2Scale)
      throw std::invalid_argument("MegaMoE requires four non-null canonical weight buffers");
    cudaStreamCaptureStatus capture;
    MSCCLPP_CUDATHROW(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("Construct MegaMoE outside CUDA Graph capture");
    MSCCLPP_CUDATHROW(cudaGetDevice(&p.device));
    p.config = c;
    if (kernelPath.empty()) {
      if (!kernelId.empty() && kernelId != "builtin")
        throw std::invalid_argument("MegaMoE kernel_id requires an explicit kernel_path");
      p.layout = getSymmetricLayout(c);
      p.workspaceBytes = getPrivateWorkspaceBytes(c);
      p.resources = preflightKernel(c);
    } else {
      // Even rejected modules must stay mapped until Impl synchronizes: CUDA
      // may still have deferred registration work referencing their host stubs.
      p.module = std::make_shared<JitModule>();
      p.module->load(kernelPath, kernelId);
      p.kernelId = kernelId;
      const auto config = jitConfig(c);
      MegaMoeJitLayoutV1 layout{};
      char error[ErrorCapacity]{};
      checkJit(p.module->api->preflight(&config, &layout, error, sizeof(error)), error, "preflight");
      p.layout = SymmetricLayout{size_t(layout.symmetricBytes), size_t(layout.input),
                                 size_t(layout.topkIds),        size_t(layout.topkWeights),
                                 size_t(layout.partialOutput),  size_t(layout.epoch),
                                 size_t(layout.peerSignals),    size_t(layout.expectedPeerSignals),
                                 size_t(layout.tokenCount)};
      p.workspaceBytes = size_t(layout.privateBytes);
      p.resources = KernelResources{layout.ctaCount, p.device, size_t(layout.sharedBytes)};
      if (!p.layout.bytes || !p.workspaceBytes || !p.resources.sharedBytes || p.resources.ctas < detail::ClusterM ||
          p.resources.ctas % detail::ClusterM || layout.reserved)
        throw std::invalid_argument("MegaMoE JIT preflight returned invalid workspace or occupancy metadata");
    }
    info = rankInfo(c, p.device);
    info.tag = tag;
    info.specialization = {kernelTileM(), kernelTileN(), kernelTileK(), kernelLoadStages(), kernelTransformStages()};
    std::memcpy(info.kernelId, p.kernelId.c_str(), p.kernelId.size() + 1);
    info.symmetricBytes = p.layout.bytes;
    info.privateBytes = p.workspaceBytes;
  });
  validatePeers(comm, c, info);
  p.communicator = comm;
  PackedWeights packed{};
  collectiveCheck(comm, "allocation and weight packing", [&] {
    p.symmetric = std::make_unique<GpuBuffer<char>>(p.layout.bytes);
    p.workspace = mscclpp::detail::gpuCallocShared<char>(p.workspaceBytes);
    p.peerBases = mscclpp::detail::gpuCallocShared<uint64_t>(c.worldSize);
    p.startSignal = mscclpp::detail::gpuCallocShared<uint32_t>(1);
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&p.startResetEvent, cudaEventDisableTiming));
    p.localMemory = p.communicator->registerMemory(p.symmetric->data(), p.layout.bytes, Transport::CudaIpc);
    size_t localExperts = c.numExperts / c.worldSize;
    const std::array<size_t, 4> sizes{product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden)}),
                                      product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden / 32)}),
                                      product({localExperts, size_t(c.hidden), size_t(c.intermediate)}),
                                      product({localExperts, size_t(c.hidden), size_t(c.intermediate / 32)})};
    for (size_t i = 0; i < sizes.size(); ++i) p.weightBuffers[i] = mscclpp::detail::gpuCallocShared<uint8_t>(sizes[i]);
    packed = PackedWeights{p.weightBuffers[0].get(), p.weightBuffers[1].get(), p.weightBuffers[2].get(),
                           p.weightBuffers[3].get()};
    if (p.module) {
      const auto config = jitConfig(c);
      const auto source = jitWeights(weights);
      const auto destination = jitWeights(packed);
      char error[ErrorCapacity]{};
      checkJit(p.module->api->packWeights(&config, &source, &destination, stream, error, sizeof(error)), error,
               "weight packing");
    } else {
      packNativeWeights(c, weights, packed, stream);
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  });
  std::vector<std::shared_future<RegisteredMemory>> memoryFutures;
  std::vector<std::shared_future<Connection>> connectionFutures;
  std::vector<uint64_t> peerPointers;
  collectiveCheck(comm, "peer exchange setup", [&] {
    p.peerMemories.resize(c.worldSize);
    p.peerMemories[c.rank] = p.localMemory;
    memoryFutures.resize(c.worldSize);
    connectionFutures.resize(c.worldSize);
    peerPointers.resize(c.worldSize);
    for (int rank = 0; rank < c.worldSize; ++rank) {
      if (rank == c.rank) continue;
      connectionFutures[rank] = p.communicator->connect(Transport::CudaIpc, rank, tag);
      p.communicator->sendMemory(p.localMemory, rank, tag + 1);
      memoryFutures[rank] = p.communicator->recvMemory(rank, tag + 1);
    }
  });
  collectiveCheck(comm, "peer mapping", [&] {
    for (int rank = 0; rank < c.worldSize; ++rank) {
      if (rank != c.rank) {
        p.connections.push_back(connectionFutures[rank].get());
        p.peerMemories[rank] = memoryFutures[rank].get();
      }
      if (!p.peerMemories[rank].data() || p.peerMemories[rank].size() != p.layout.bytes)
        throw std::runtime_error("MegaMoE could not map a matching CudaIpc peer workspace");
      peerPointers[rank] = reinterpret_cast<uint64_t>(p.peerMemories[rank].data());
    }
  });
  collectiveCheck(comm, "plan creation", [&] {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(p.peerBases.get(), peerPointers.data(), c.worldSize * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    if (p.module) {
      p.jitPlan = std::make_unique<JitPlan>(p.module);
      const auto config = jitConfig(c);
      const auto weights = jitWeights(packed);
      char error[ErrorCapacity]{};
      checkJit(p.module->api->createPlan(&config, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), &weights,
                                         &p.jitPlan->handle, error, sizeof(error)),
               error, "plan creation");
      if (!p.jitPlan->handle) throw std::runtime_error("MegaMoE JIT module returned a null plan");
    } else {
      p.plan = createKernelPlan(c, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), packed);
      p.resources.ctas = kernelPlanCtaCount(*p.plan);
      p.resources.sharedBytes = kernelPlanSharedBytes(*p.plan);
    }
  });
}

MegaMoeContext::~MegaMoeContext() = default;
const NativeConfig& MegaMoeContext::config() const { return impl_->config; }
int MegaMoeContext::device() const { return impl_->device; }
void* MegaMoeContext::input() const { return impl_->symmetric->data() + impl_->layout.input; }
void* MegaMoeContext::topkIds() const { return impl_->symmetric->data() + impl_->layout.topkIds; }
void* MegaMoeContext::topkWeights() const { return impl_->symmetric->data() + impl_->layout.topkWeights; }
int MegaMoeContext::ctaCount() const { return impl_->resources.ctas; }
size_t MegaMoeContext::sharedBytes() const { return impl_->resources.sharedBytes; }
size_t MegaMoeContext::symmetricBytes() const { return impl_->layout.bytes; }
size_t MegaMoeContext::privateBytes() const { return impl_->workspaceBytes; }
const std::string& MegaMoeContext::kernelId() const { return impl_->kernelId; }
int MegaMoeContext::kernelTileM() const { return impl_->module ? impl_->module->api->tileM : detail::TileM; }
int MegaMoeContext::kernelTileN() const { return impl_->module ? impl_->module->api->tileN : detail::TileN; }
int MegaMoeContext::kernelTileK() const { return impl_->module ? impl_->module->api->tileK : detail::TileK; }
int MegaMoeContext::kernelLoadStages() const {
  return impl_->module ? impl_->module->api->loadStages : detail::LoadStages;
}
int MegaMoeContext::kernelTransformStages() const {
  return impl_->module ? impl_->module->api->transformStages : detail::TransformStages;
}

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
  auto* signal = signalStart ? impl_->startSignal.get() : nullptr;
  if (impl_->jitPlan) {
    char error[ErrorCapacity]{};
    checkJit(
        impl_->module->api->launch(impl_->jitPlan->handle, tokens, output, stream, signal, 0, error, sizeof(error)),
        error, "launch");
  } else {
    launchNativeMegaMoe(impl_->plan, tokens, output, stream, signal);
  }
  impl_->startRecorded = signalStart;
}

void MegaMoeContext::forwardShared(const void* x, void* output, int tokens, cudaStream_t stream) {
  if (config().worldSize != 1 || config().numExperts != 1 || config().topK != 1)
    throw std::invalid_argument("Shared forward requires world_size=1, num_experts=1, top_k=1");
  const size_t inputBytes = validateForward(x, output, tokens);
  impl_->startRecorded = false;
  stage(input(), x, inputBytes, stream);
  if (impl_->jitPlan) {
    char error[ErrorCapacity]{};
    checkJit(
        impl_->module->api->launch(impl_->jitPlan->handle, tokens, output, stream, nullptr, 1, error, sizeof(error)),
        error, "shared launch");
  } else {
    launchNativeSharedExpert(impl_->plan, tokens, output, stream);
  }
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
