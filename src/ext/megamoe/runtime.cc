// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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
  return MegaMoeJitConfigV1{c.rank,
                            c.worldSize,
                            c.maxTokens,
                            c.hidden,
                            c.intermediate,
                            c.numExperts,
                            c.topK,
                            c.smMargin,
                            c.gateUpClamp,
                            static_cast<int32_t>(c.activationDtype),
                            c.activationScale,
                            int32_t(c.combineRouteWeights)};
}

MegaMoeJitWeightsV1 jitWeights(const PackedWeights& weights) {
  return MegaMoeJitWeightsV1{weights.fc1, weights.fc1Scale, weights.fc2, weights.fc2Scale};
}

bool validW4Metadata(const MegaMoeJitApiV1& api) {
  const bool tileN = api.tileN == 128;
  const bool tileK = api.tileK == 128 || api.tileK == 256 || api.tileK == 512;
  const bool transfer = api.transferRegisters == 32 || api.transferRegisters == 64 || api.transferRegisters == 96 ||
                        api.transferRegisters == 128;
  const bool dispatchChunk = api.dispatchChunk == 512 || api.dispatchChunk == 1024 || api.dispatchChunk == 2048 ||
                             api.dispatchChunk == 3072 || api.dispatchChunk == 4096;
  const int64_t threads = int64_t(api.numWarps) * 32;
  const int64_t epilogueThreads = int64_t(api.epilogueWarps) * 32;
  const int64_t lowRegisterThreads = threads - epilogueThreads - 128;
  const int64_t registerUse =
      epilogueThreads * api.epilogueRegisters + int64_t(128) * api.transferRegisters + lowRegisterThreads * 32;
  return (api.clusterSize == 1 || api.clusterSize == 2) && api.tileM == 128 * api.clusterSize && tileN && tileK &&
         api.loadStages >= 2 && api.loadStages <= 10 && (api.numWarps == 12 || api.numWarps == 16) && transfer &&
         (api.loadWarps == 1 || api.loadWarps == 2) && (api.splitPipelines == 0 || api.splitPipelines == 1) &&
         (!api.splitPipelines || api.loadWarps == 2) && (api.epilogueTokens == 16 || api.epilogueTokens == 32) &&
         api.tileN % api.epilogueTokens == 0 && (api.epilogueWarps == 4 || api.epilogueWarps == 8) &&
         api.epilogueRegisters >= 128 && api.epilogueRegisters <= 224 && api.epilogueRegisters % 8 == 0 &&
         dispatchChunk && api.dispatchChunk % api.tileK == 0 &&
         (api.dispatchWarps == 2 || api.dispatchWarps == 4 || api.dispatchWarps == 8) &&
         (api.dispatchStages == 1 || api.dispatchStages == 2) &&
         api.epilogueWarps + 4 + api.dispatchWarps <= api.numWarps && lowRegisterThreads >= 0 &&
         registerUse <= threads * 128;
}

struct JitModule {
  void* handle = nullptr;
  const MegaMoeJitApiV1* api = nullptr;
  MegaMoeJitLaunchDirectV1 launchDirect = nullptr;
  bool fixedTokenCount = false;
  uint32_t optimizationFlags = 0;

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
        api->layoutBytes != sizeof(MegaMoeJitLayoutV1) || api->accumulatorStages != 2 || api->architecture != 1000)
      throw std::invalid_argument("MegaMoE JIT module metadata does not match the native ABI");
    if (std::memcmp(api->kernelId, id.c_str(), MSCCLPP_MEGAMOE_JIT_ID_CAPACITY))
      throw std::invalid_argument("MegaMoE JIT module kernel_id does not match the requested specialization");
    if (!validW4Metadata(*api))
      throw std::invalid_argument("MegaMoE JIT module declares an unsupported specialization policy");
    (void)dlerror();
    auto routing = reinterpret_cast<int (*)()>(dlsym(handle, MSCCLPP_MEGAMOE_JIT_FIXED_TOKEN_COUNT_ENTRYPOINT));
    const char* routingError = dlerror();
    if (routingError || !routing)
      throw std::invalid_argument("MegaMoE W4A8 JIT module is missing routing policy metadata; recompile it");
    int fixed = routing();
    if (fixed != 0 && fixed != 1)
      throw std::invalid_argument("MegaMoE W4A8 JIT module declares an invalid fixed-token policy");
    fixedTokenCount = bool(fixed);
    (void)dlerror();
    auto optimizations =
        reinterpret_cast<uint32_t (*)()>(dlsym(handle, MSCCLPP_MEGAMOE_JIT_OPTIMIZATION_FLAGS_ENTRYPOINT));
    const char* optimizationError = dlerror();
    if (optimizationError || !optimizations)
      throw std::invalid_argument("MegaMoE W4A8 JIT module is missing optimization policy metadata; recompile it");
    optimizationFlags = optimizations();
    if (optimizationFlags & ~(MSCCLPP_MEGAMOE_JIT_EXPERT_MAJOR_TASKS | MSCCLPP_MEGAMOE_JIT_CACHE_WEIGHTS |
                              MSCCLPP_MEGAMOE_JIT_ADAPTIVE_CACHE_WEIGHTS | MSCCLPP_MEGAMOE_JIT_FUSE_STATIC_QUANTIZE))
      throw std::invalid_argument("MegaMoE W4A8 JIT module declares unsupported optimization flags");
    if ((optimizationFlags & MSCCLPP_MEGAMOE_JIT_CACHE_WEIGHTS) &&
        (optimizationFlags & MSCCLPP_MEGAMOE_JIT_ADAPTIVE_CACHE_WEIGHTS))
      throw std::invalid_argument("MegaMoE W4A8 static and adaptive weight caching are mutually exclusive");
    (void)dlerror();
    launchDirect =
        reinterpret_cast<MegaMoeJitLaunchDirectV1>(dlsym(handle, MSCCLPP_MEGAMOE_JIT_LAUNCH_DIRECT_ENTRYPOINT));
    const char* launchError = dlerror();
    if (launchError || !launchDirect)
      throw std::invalid_argument("MegaMoE W4A8 JIT module is missing the direct launch entrypoint; recompile it");
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
  std::array<int, 10> config;
  int tag = 0;
  std::array<int, 20> specialization;
  uint64_t sharedOwner = 0;
  uint64_t newOwner = 0;
  char kernelId[MSCCLPP_MEGAMOE_JIT_ID_CAPACITY]{};
  size_t symmetricBytes = 0;
  size_t privateBytes = 0;
  float gateUpClamp;
  float activationScale;
  int sharedHidden = 0;
  int sharedIntermediate = 0;
  float sharedGateUpClamp = -1.0f;
  bool fusedShared = false;
  char gpuUuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
  unsigned char fabricUuid[NVML_GPU_FABRIC_UUID_LEN]{};
  unsigned int cliqueId = 0;
  bool fabricReady = false;
};

RankInfo rankInfo(const NativeConfig& c, const SharedExpertConfig* shared, int device) {
  RankInfo info{};
  info.config = {c.worldSize, c.maxTokens, c.hidden,           c.intermediate,         c.numExperts,
                 c.topK,      c.smMargin,  int(c.weightDtype), int(c.activationDtype), int(c.combineRouteWeights)};
  info.gateUpClamp = c.gateUpClamp;
  info.activationScale = c.activationScale;
  if (shared) {
    info.sharedHidden = shared->hidden ? shared->hidden : c.hidden;
    info.sharedIntermediate = shared->intermediate;
    info.sharedGateUpClamp = shared->gateUpClamp;
    info.fusedShared = true;
  }
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

uint64_t validatePeers(const std::shared_ptr<Communicator>& comm, const NativeConfig& c, const RankInfo& info) {
  std::vector<RankInfo> peers(c.worldSize);
  peers[c.rank] = info;
  comm->bootstrap()->allGather(peers.data(), sizeof(RankInfo));
  const auto& local = peers[c.rank];
  for (int rank = 0; rank < c.worldSize; ++rank) {
    const auto& peer = peers[rank];
    if (peer.config != local.config || peer.gateUpClamp != local.gateUpClamp ||
        peer.activationScale != local.activationScale || peer.sharedHidden != local.sharedHidden ||
        peer.sharedIntermediate != local.sharedIntermediate || peer.sharedGateUpClamp != local.sharedGateUpClamp ||
        peer.fusedShared != local.fusedShared || peer.tag != local.tag)
      throw std::invalid_argument("MegaMoE configurations and bootstrap tags must match on every rank (except rank)");
    if (peer.sharedOwner != local.sharedOwner)
      throw std::invalid_argument(
          "MegaMoE shared_workspace mode and original workspace owner must match on every rank");
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
  return peers[0].newOwner;
}

auto configSignature(const NativeConfig& c) {
  return std::tie(c.rank, c.worldSize, c.maxTokens, c.hidden, c.intermediate, c.numExperts, c.topK, c.smMargin,
                  c.gateUpClamp, c.weightDtype, c.activationDtype, c.activationScale, c.combineRouteWeights);
}

auto layoutSignature(const SymmetricLayout& l) {
  return std::tie(l.bytes, l.input, l.topkIds, l.topkWeights, l.partialOutput, l.epoch, l.peerSignals,
                  l.expectedPeerSignals, l.tokenCount, l.quantizedInput, l.quantizedInputScale, l.quantizedInputReady,
                  l.routedInputReady, l.routingHeaders, l.routingPackets);
}

std::array<int, 20> specialization(const MegaMoeContext& c) {
  return {c.kernelTileM(),
          c.kernelTileN(),
          c.kernelTileK(),
          c.kernelLoadStages(),
          c.kernelTransformStages(),
          c.kernelNumWarps(),
          c.kernelTransferRegisters(),
          c.kernelLoadWarps(),
          int(c.kernelSplitPipelines()),
          c.kernelEpilogueTokens(),
          c.kernelEpilogueWarps(),
          c.kernelEpilogueRegisters(),
          c.kernelDispatchChunk(),
          c.kernelDispatchWarps(),
          c.kernelDispatchStages(),
          int(c.kernelExpertMajorTasks()),
          int(c.kernelCacheWeights()),
          int(c.kernelAdaptiveCacheWeights()),
          c.kernelClusterSize(),
          int(c.kernelFixedTokenCount())};
}

bool overlaps(const void* a, size_t aBytes, const void* b, size_t bBytes) {
  auto x = reinterpret_cast<uintptr_t>(a);
  auto y = reinterpret_cast<uintptr_t>(b);
  return aBytes && bBytes && (x <= y ? y - x < aBytes : x - y < bBytes);
}

void validateStage(void* destination, const void* source, size_t bytes) {
  if (!bytes || source == destination) return;
  if (!source || overlaps(source, bytes, destination, bytes))
    throw std::invalid_argument("MegaMoE input must be non-null and not partially overlap its staging buffer");
}

void stage(void* destination, const void* source, size_t bytes, cudaStream_t stream) {
  validateStage(destination, source, bytes);
  if (!bytes || source == destination) return;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
}

struct RuntimeWorkspace {
  int device = -1;
  uint64_t owner = 0;
  std::shared_ptr<Communicator> communicator;
  std::unique_ptr<GpuBuffer<char>> symmetric;
  std::shared_ptr<char> workspace;
  std::shared_ptr<uint64_t> peerBases;
  std::shared_ptr<uint32_t> kernelEntrySignal;
  cudaEvent_t startResetEvent = nullptr;
  bool startRecorded = false;
  RegisteredMemory localMemory;
  std::vector<RegisteredMemory> peerMemories;
  std::vector<Connection> connections;

  ~RuntimeWorkspace() {
    if (device < 0) return;
    // Keep mappings alive until pending work completes, including when the last
    // alias (rather than the original context) releases this runtime.
    int original = -1;
    (void)cudaGetDevice(&original);
    (void)cudaSetDevice(device);
    (void)cudaDeviceSynchronize();
    if (startResetEvent) (void)cudaEventDestroy(startResetEvent);
    connections.clear();
    peerMemories.clear();
    localMemory = RegisteredMemory{};
    kernelEntrySignal.reset();
    peerBases.reset();
    workspace.reset();
    symmetric.reset();
    if (original >= 0) (void)cudaSetDevice(original);
  }
};

}  // namespace

struct MegaMoeContext::Impl {
  NativeConfig config;
  std::optional<SharedExpertConfig> sharedConfig;
  std::string kernelId = "builtin";
  KernelResources resources;
  int device = -1;
  SymmetricLayout layout;
  size_t workspaceBytes = 0;
  std::shared_ptr<RuntimeWorkspace> runtime;
  std::array<std::shared_ptr<uint8_t>, 4> weightBuffers;
  std::array<std::shared_ptr<uint8_t>, 4> sharedWeightBuffers;
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
    plan.reset();
    jitPlan.reset();
    module.reset();
    weightBuffers = {};
    sharedWeightBuffers = {};
    runtime.reset();
    if (original >= 0) (void)cudaSetDevice(original);
  }
};

MegaMoeContext::MegaMoeContext(std::shared_ptr<Communicator> comm, const NativeConfig& c, const PackedWeights& weights,
                               cudaStream_t stream, int tag, const std::string& kernelPath, const std::string& kernelId,
                               std::shared_ptr<MegaMoeContext> sharedWorkspace)
    : MegaMoeContext(std::move(comm), c, weights, nullptr, nullptr, stream, tag, kernelPath, kernelId,
                     std::move(sharedWorkspace)) {}

MegaMoeContext::MegaMoeContext(std::shared_ptr<Communicator> comm, const NativeConfig& c, const PackedWeights& weights,
                               const SharedExpertConfig& sharedConfig, const PackedWeights& sharedWeights,
                               cudaStream_t stream, int tag, std::shared_ptr<MegaMoeContext> sharedWorkspace)
    : MegaMoeContext(std::move(comm), c, weights, &sharedConfig, &sharedWeights, stream, tag, "", "",
                     std::move(sharedWorkspace)) {}

MegaMoeContext::MegaMoeContext(std::shared_ptr<Communicator> comm, const NativeConfig& c, const PackedWeights& weights,
                               const SharedExpertConfig* sharedConfig, const PackedWeights* sharedWeights,
                               cudaStream_t stream, int tag, const std::string& kernelPath, const std::string& kernelId,
                               std::shared_ptr<MegaMoeContext> sharedWorkspace)
    : impl_(std::make_unique<Impl>()) {
  if (!comm || !comm->bootstrap()) throw std::invalid_argument("MegaMoE requires a communicator with a bootstrap");
  auto& p = *impl_;
  RankInfo info{};
  collectiveCheck(comm, "preflight", [&] {
    if (bool(sharedConfig) != bool(sharedWeights))
      throw std::invalid_argument("MegaMoE fused construction requires both shared configuration and weights");
    if (sharedConfig)
      validateFusedConfig(c, *sharedConfig);
    else
      validateNativeConfig(c);
    if (comm->bootstrap()->getRank() != c.rank || comm->bootstrap()->getNranks() != c.worldSize)
      throw std::invalid_argument("MegaMoE rank/worldSize must match its communicator");
    if (tag < 0 || tag == std::numeric_limits<int>::max())
      throw std::invalid_argument("MegaMoE bootstrap tag must be in [0, INT_MAX-1)");
    if (!weights.fc1 || !weights.fc1Scale || !weights.fc2 || !weights.fc2Scale ||
        (sharedWeights &&
         (!sharedWeights->fc1 || !sharedWeights->fc1Scale || !sharedWeights->fc2 || !sharedWeights->fc2Scale)))
      throw std::invalid_argument("MegaMoE requires four non-null canonical weight buffers");
    cudaStreamCaptureStatus capture;
    MSCCLPP_CUDATHROW(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("Construct MegaMoE outside CUDA Graph capture");
    MSCCLPP_CUDATHROW(cudaGetDevice(&p.device));
    p.config = c;
    if (sharedConfig) p.sharedConfig = *sharedConfig;
    if (kernelPath.empty()) {
      if (!kernelId.empty() && kernelId != "builtin")
        throw std::invalid_argument("MegaMoE kernel_id requires an explicit kernel_path");
      p.layout = getSymmetricLayout(c);
      p.workspaceBytes = sharedConfig ? getFusedPrivateWorkspaceBytes(c, *sharedConfig) : getPrivateWorkspaceBytes(c);
      p.resources = sharedConfig ? preflightFusedKernel(c, *sharedConfig) : preflightKernel(c);
    } else {
      if (sharedConfig) throw std::invalid_argument("MegaMoE fused shared mode does not support external JIT modules");
      if (c.weightDtype != WeightDtype::Mxfp4)
        throw std::invalid_argument("MegaMoE W4A8 JIT module requires weightDtype=Mxfp4");
      // Even rejected modules must stay mapped until Impl synchronizes: CUDA
      // may still have deferred registration work referencing their host stubs.
      p.module = std::make_shared<JitModule>();
      p.module->load(kernelPath, kernelId);
      p.kernelId = kernelId;
      const auto config = jitConfig(c);
      MegaMoeJitLayoutV1 layout{};
      char error[ErrorCapacity]{};
      checkJit(p.module->api->preflight(&config, &layout, error, sizeof(error)), error, "preflight");
      p.layout = SymmetricLayout{size_t(layout.symmetricBytes),
                                 size_t(layout.input),
                                 size_t(layout.topkIds),
                                 size_t(layout.topkWeights),
                                 size_t(layout.partialOutput),
                                 size_t(layout.epoch),
                                 size_t(layout.peerSignals),
                                 size_t(layout.expectedPeerSignals),
                                 size_t(layout.tokenCount),
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0};
      p.workspaceBytes = size_t(layout.privateBytes);
      p.resources = KernelResources{layout.ctaCount, p.device, size_t(layout.sharedBytes)};
      if (!p.layout.bytes || !p.workspaceBytes || !p.resources.sharedBytes ||
          p.resources.ctas < p.module->api->clusterSize || p.resources.ctas % p.module->api->clusterSize ||
          layout.reserved)
        throw std::invalid_argument("MegaMoE JIT preflight returned invalid workspace or occupancy metadata");
    }
    info = rankInfo(c, sharedConfig, p.device);
    info.tag = tag;
    info.specialization = specialization(*this);
    if (sharedWorkspace) {
      const auto& owner = *sharedWorkspace->impl_;
      if (owner.device != p.device || owner.runtime->communicator.get() != comm.get())
        throw std::invalid_argument("MegaMoE shared_workspace must use the same CUDA device and communicator");
      if (configSignature(owner.config) != configSignature(c))
        throw std::invalid_argument("MegaMoE shared_workspace must use the same native configuration");
      const bool sharedConfigMatches =
          owner.sharedConfig.has_value() == p.sharedConfig.has_value() &&
          (!p.sharedConfig || (owner.sharedConfig->hidden == p.sharedConfig->hidden &&
                               owner.sharedConfig->intermediate == p.sharedConfig->intermediate &&
                               owner.sharedConfig->gateUpClamp == p.sharedConfig->gateUpClamp));
      if (!sharedConfigMatches)
        throw std::invalid_argument("MegaMoE shared_workspace must use the same fused shared configuration");
      if (owner.kernelId != p.kernelId || specialization(*sharedWorkspace) != info.specialization ||
          layoutSignature(owner.layout) != layoutSignature(p.layout) || owner.workspaceBytes != p.workspaceBytes ||
          owner.resources.ctas != p.resources.ctas || owner.resources.sharedBytes != p.resources.sharedBytes)
        throw std::invalid_argument("MegaMoE shared_workspace must use the same kernel specialization and layout");
      info.sharedOwner = owner.runtime->owner;
    } else if (c.rank == 0) {
      // Tags may be reused after construction. A root-issued identity also
      // distinguishes independent workspaces created with the same tag/config.
      static std::atomic<uint64_t> nextOwner{1};
      info.newOwner = nextOwner.fetch_add(1, std::memory_order_relaxed);
    }
    std::memcpy(info.kernelId, p.kernelId.c_str(), p.kernelId.size() + 1);
    info.symmetricBytes = p.layout.bytes;
    info.privateBytes = p.workspaceBytes;
  });
  const uint64_t newOwner = validatePeers(comm, c, info);
  PackedWeights packed{};
  PackedWeights packedShared{};
  collectiveCheck(comm, "allocation and weight packing", [&] {
    if (sharedWorkspace) {
      p.runtime = sharedWorkspace->impl_->runtime;
    } else {
      p.runtime = std::make_shared<RuntimeWorkspace>();
      auto& r = *p.runtime;
      r.device = p.device;
      r.owner = newOwner;
      r.communicator = comm;
      r.symmetric = std::make_unique<GpuBuffer<char>>(p.layout.bytes);
      r.workspace = mscclpp::detail::gpuCallocShared<char>(p.workspaceBytes);
      r.peerBases = mscclpp::detail::gpuCallocShared<uint64_t>(c.worldSize);
      r.kernelEntrySignal = mscclpp::detail::gpuCallocShared<uint32_t>(1);
      MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&r.startResetEvent, cudaEventDisableTiming));
      r.localMemory = comm->registerMemory(r.symmetric->data(), p.layout.bytes, Transport::CudaIpc);
    }
    auto allocateWeights = [&](const NativeConfig& config, std::array<std::shared_ptr<uint8_t>, 4>& buffers) {
      size_t localExperts = config.numExperts / config.worldSize;
      const size_t valuesPerByte = config.weightDtype == WeightDtype::Mxfp4 ? 2 : 1;
      const std::array<size_t, 4> sizes{
          product({localExperts, 2, size_t(config.intermediate), size_t(config.hidden)}) / valuesPerByte,
          product({localExperts, 2, size_t(config.intermediate), size_t(config.hidden / 32)}),
          product({localExperts, size_t(config.hidden), size_t(config.intermediate)}) / valuesPerByte,
          product({localExperts, size_t(config.hidden), size_t(config.intermediate / 32)})};
      for (size_t i = 0; i < sizes.size(); ++i) buffers[i] = mscclpp::detail::gpuCallocShared<uint8_t>(sizes[i]);
      return PackedWeights{buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[3].get()};
    };
    packed = allocateWeights(c, p.weightBuffers);
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
    if (c.weightDtype == WeightDtype::Fp8Nv) {
      // fp8_nv scale buffers each hold one FP32 scale for the whole weight matrix of the single local expert.
      MSCCLPP_CUDATHROW(
          cudaMemcpyAsync(&packed.fc1TensorScale, weights.fc1Scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
      MSCCLPP_CUDATHROW(
          cudaMemcpyAsync(&packed.fc2TensorScale, weights.fc2Scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    if (sharedConfig) {
      NativeConfig local{};
      local.maxTokens = c.maxTokens;
      local.hidden = sharedConfig->hidden ? sharedConfig->hidden : c.hidden;
      local.intermediate = sharedConfig->intermediate;
      local.numExperts = 1;
      local.topK = 1;
      local.smMargin = c.smMargin;
      local.gateUpClamp = sharedConfig->gateUpClamp;
      local.weightDtype = WeightDtype::Fp8Nv;
      local.activationDtype = ActivationDtype::Fp8Static;
      local.activationScale = c.activationScale;
      packedShared = allocateWeights(local, p.sharedWeightBuffers);
      packNativeWeights(local, *sharedWeights, packedShared, stream);
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(&packedShared.fc1TensorScale, sharedWeights->fc1Scale, sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(&packedShared.fc2TensorScale, sharedWeights->fc2Scale, sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    for (float scale : {packed.fc1TensorScale, packed.fc2TensorScale})
      if (!std::isfinite(scale) || scale <= 0.0f)
        throw std::invalid_argument("MegaMoE fp8_nv weight scales must be finite positive FP32 values");
    if (sharedConfig)
      for (float scale : {packedShared.fc1TensorScale, packedShared.fc2TensorScale})
        if (!std::isfinite(scale) || scale <= 0.0f)
          throw std::invalid_argument("MegaMoE shared fp8_nv weight scales must be finite positive FP32 values");
  });
  auto& r = *p.runtime;
  if (!sharedWorkspace) {
    std::vector<std::shared_future<RegisteredMemory>> memoryFutures;
    std::vector<std::shared_future<Connection>> connectionFutures;
    std::vector<uint64_t> peerPointers;
    collectiveCheck(comm, "peer exchange setup", [&] {
      r.peerMemories.resize(c.worldSize);
      r.peerMemories[c.rank] = r.localMemory;
      memoryFutures.resize(c.worldSize);
      connectionFutures.resize(c.worldSize);
      peerPointers.resize(c.worldSize);
      for (int rank = 0; rank < c.worldSize; ++rank) {
        if (rank == c.rank) continue;
        connectionFutures[rank] = comm->connect(Transport::CudaIpc, rank, tag);
        comm->sendMemory(r.localMemory, rank, tag + 1);
        memoryFutures[rank] = comm->recvMemory(rank, tag + 1);
      }
    });
    collectiveCheck(comm, "peer mapping", [&] {
      for (int rank = 0; rank < c.worldSize; ++rank) {
        if (rank != c.rank) {
          r.connections.push_back(connectionFutures[rank].get());
          r.peerMemories[rank] = memoryFutures[rank].get();
        }
        if (!r.peerMemories[rank].data() || r.peerMemories[rank].size() != p.layout.bytes)
          throw std::runtime_error("MegaMoE could not map a matching CudaIpc peer workspace");
        peerPointers[rank] = reinterpret_cast<uint64_t>(r.peerMemories[rank].data());
      }
    });
    collectiveCheck(comm, "peer address upload", [&] {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(r.peerBases.get(), peerPointers.data(), c.worldSize * sizeof(uint64_t),
                                        cudaMemcpyHostToDevice, stream));
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    });
  }
  collectiveCheck(comm, "plan creation", [&] {
    // Plan creation builds only host-side, weight-specific descriptors. It must
    // not reset workspace counters or the device-resident epoch of an alias.
    if (p.module) {
      p.jitPlan = std::make_unique<JitPlan>(p.module);
      const auto config = jitConfig(c);
      const auto weights = jitWeights(packed);
      char error[ErrorCapacity]{};
      checkJit(p.module->api->createPlan(&config, r.symmetric->data(), r.peerBases.get(), r.workspace.get(), &weights,
                                         &p.jitPlan->handle, error, sizeof(error)),
               error, "plan creation");
      if (!p.jitPlan->handle) throw std::runtime_error("MegaMoE JIT module returned a null plan");
    } else {
      p.plan = sharedConfig ? createFusedKernelPlan(c, *sharedConfig, r.symmetric->data(), r.peerBases.get(),
                                                    r.workspace.get(), packed, packedShared)
                            : createKernelPlan(c, r.symmetric->data(), r.peerBases.get(), r.workspace.get(), packed);
      p.resources.ctas = kernelPlanCtaCount(*p.plan);
      p.resources.sharedBytes = kernelPlanSharedBytes(*p.plan);
    }
  });
}

MegaMoeContext::~MegaMoeContext() = default;
const NativeConfig& MegaMoeContext::config() const { return impl_->config; }
int MegaMoeContext::device() const { return impl_->device; }
void* MegaMoeContext::input() const { return impl_->runtime->symmetric->data() + impl_->layout.input; }
void* MegaMoeContext::topkIds() const { return impl_->runtime->symmetric->data() + impl_->layout.topkIds; }
void* MegaMoeContext::topkWeights() const { return impl_->runtime->symmetric->data() + impl_->layout.topkWeights; }
int MegaMoeContext::ctaCount() const { return impl_->resources.ctas; }
size_t MegaMoeContext::sharedBytes() const { return impl_->resources.sharedBytes; }
size_t MegaMoeContext::symmetricBytes() const { return impl_->layout.bytes; }
size_t MegaMoeContext::privateBytes() const { return impl_->workspaceBytes; }
uintptr_t MegaMoeContext::workspaceId() const { return reinterpret_cast<uintptr_t>(impl_->runtime->workspace.get()); }
std::array<uintptr_t, 4> MegaMoeContext::packedWeightPointers() const {
  std::array<uintptr_t, 4> pointers{};
  for (size_t i = 0; i < pointers.size(); ++i) pointers[i] = reinterpret_cast<uintptr_t>(impl_->weightBuffers[i].get());
  return pointers;
}
const std::string& MegaMoeContext::kernelId() const { return impl_->kernelId; }
namespace {
// Builtin W8A8 local experts run the W4A8 pipeline specialization.
bool usesW4Pipeline(const NativeConfig& c) {
  return c.weightDtype == WeightDtype::Mxfp4 || c.activationDtype == ActivationDtype::Fp8Static;
}
}  // namespace

int MegaMoeContext::kernelTileM() const {
  if (impl_->module) return impl_->module->api->tileM;
  return detail::W4TileM;
}
int MegaMoeContext::kernelClusterSize() const {
  if (impl_->module) return impl_->module->api->clusterSize;
  return usesW4Pipeline(config()) ? detail::W4ClusterSize : 2;
}
int MegaMoeContext::kernelTileN() const {
  if (impl_->module) return impl_->module->api->tileN;
  return detail::W4TileN;
}
int MegaMoeContext::kernelTileK() const {
  if (impl_->module) return impl_->module->api->tileK;
  return detail::W4TileK;
}
int MegaMoeContext::kernelLoadStages() const {
  if (impl_->module) return impl_->module->api->loadStages;
  return usesW4Pipeline(config()) ? detail::W4LoadStages : detail::W8LoadStages;
}
int MegaMoeContext::kernelTransformStages() const { return usesW4Pipeline(config()) ? 0 : detail::W8TransformStages; }
int MegaMoeContext::kernelNumWarps() const {
  if (impl_->module) return impl_->module->api->numWarps;
  return usesW4Pipeline(config()) ? detail::W4NumWarps : detail::W8A16WarpSchedule<false>::NumWarps;
}
int MegaMoeContext::kernelTransferRegisters() const {
  if (impl_->module) return impl_->module->api->transferRegisters;
  return usesW4Pipeline(config()) ? detail::W4TransferRegisters : 32;
}
int MegaMoeContext::kernelLoadWarps() const {
  if (impl_->module) return impl_->module->api->loadWarps;
  return usesW4Pipeline(config()) ? detail::W4LoadWarps : 2;
}
bool MegaMoeContext::kernelSplitPipelines() const {
  if (impl_->module) return bool(impl_->module->api->splitPipelines);
  return usesW4Pipeline(config()) ? detail::W4SplitPipelines : false;
}
int MegaMoeContext::kernelEpilogueTokens() const {
  if (impl_->module) return impl_->module->api->epilogueTokens;
  return usesW4Pipeline(config()) ? detail::W4EpilogueTokens : 32;
}
int MegaMoeContext::kernelEpilogueWarps() const {
  if (impl_->module) return impl_->module->api->epilogueWarps;
  return usesW4Pipeline(config()) ? detail::W4EpilogueWarps : 4;
}
int MegaMoeContext::kernelEpilogueRegisters() const {
  if (impl_->module) return impl_->module->api->epilogueRegisters;
  return usesW4Pipeline(config()) ? detail::W4EpilogueRegisters : 224;
}
int MegaMoeContext::kernelDispatchChunk() const {
  if (impl_->module) return impl_->module->api->dispatchChunk;
  return detail::W4DispatchChunk;
}
int MegaMoeContext::kernelDispatchWarps() const {
  if (impl_->module) return impl_->module->api->dispatchWarps;
  return usesW4Pipeline(config()) ? detail::W4DispatchWarps : 4;
}
int MegaMoeContext::kernelDispatchStages() const {
  if (impl_->module) return impl_->module->api->dispatchStages;
  return usesW4Pipeline(config()) ? detail::W4DispatchStages : 2;
}
bool MegaMoeContext::kernelFixedTokenCount() const {
  if (impl_->module) return impl_->module->fixedTokenCount;
  return usesW4Pipeline(config()) ? detail::W4FixedTokenCount : false;
}
bool MegaMoeContext::kernelExpertMajorTasks() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_EXPERT_MAJOR_TASKS) != 0;
  return usesW4Pipeline(config()) ? detail::W4ExpertMajorTasks : false;
}
bool MegaMoeContext::kernelCacheWeights() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_CACHE_WEIGHTS) != 0;
  return usesW4Pipeline(config()) ? detail::W4CacheWeights : false;
}
bool MegaMoeContext::kernelAdaptiveCacheWeights() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_ADAPTIVE_CACHE_WEIGHTS) != 0;
  return usesW4Pipeline(config()) ? detail::W4AdaptiveCacheWeights : false;
}
bool MegaMoeContext::kernelFuseStaticQuantize() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_FUSE_STATIC_QUANTIZE) != 0;
  return usesW4Pipeline(config()) ? detail::W4FuseStaticQuantize : false;
}

size_t MegaMoeContext::validateForward(const void* x, void* output, int tokens) const {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE forward must use the context's CUDA device");
  if (tokens < 0 || tokens > config().maxTokens)
    throw std::invalid_argument("MegaMoE numTokens exceeds the configured capacity");
  size_t inputBytes = size_t(tokens) * config().hidden * 2;
  if ((tokens && (!x || !output)) ||
      overlaps(output, inputBytes, impl_->runtime->symmetric->data(), impl_->layout.bytes))
    throw std::invalid_argument("MegaMoE requires non-null arrays and output disjoint from the registered workspace");
  return inputBytes;
}

void MegaMoeContext::forward(const void* x, const int32_t* ids, const float* scores, void* output, int tokens,
                             cudaStream_t stream, bool signalStart) {
  forwardImpl(x, nullptr, ids, scores, output, nullptr, tokens, stream, signalStart, false);
}

void MegaMoeContext::forwardFusedShared(const void* x, const void* sharedX, const int32_t* ids, const float* scores,
                                        void* output, void* sharedOutput, int tokens, cudaStream_t stream,
                                        bool signalStart) {
  if (!impl_->sharedConfig || !impl_->plan)
    throw std::invalid_argument("MegaMoE fused shared forward requires a context constructed with shared weights");
  const size_t routedBytes = validateForward(x, output, tokens);
  const int sharedHidden = impl_->sharedConfig->hidden ? impl_->sharedConfig->hidden : config().hidden;
  const size_t sharedBytes = size_t(tokens) * sharedHidden * sizeof(__bfloat16);
  if ((tokens && (!sharedX || !sharedOutput)) ||
      overlaps(sharedOutput, sharedBytes, impl_->runtime->symmetric->data(), impl_->layout.bytes))
    throw std::invalid_argument(
        "MegaMoE fused shared requires non-null shared arrays and output disjoint from the registered workspace");
  if (sharedHidden == config().hidden && sharedX != x)
    throw std::invalid_argument("MegaMoE equal-width fused paths must share the same input allocation");
  if (tokens && (reinterpret_cast<uintptr_t>(sharedX) & 15))
    throw std::invalid_argument("MegaMoE fused shared input must be 16-byte aligned");
  if (overlaps(output, routedBytes, sharedOutput, sharedBytes))
    throw std::invalid_argument("MegaMoE routed and shared outputs must not overlap");
  forwardImpl(x, sharedX, ids, scores, output, sharedOutput, tokens, stream, signalStart, true);
}

void MegaMoeContext::forwardImpl(const void* x, const void* sharedX, const int32_t* ids, const float* scores,
                                 void* output, void* sharedOutput, int tokens, cudaStream_t stream, bool signalStart,
                                 bool fused) {
  const size_t inputBytes = validateForward(x, output, tokens);
  const size_t routingBytes = size_t(tokens) * config().topK * 4;
  if (tokens && (!ids || !scores)) throw std::invalid_argument("MegaMoE routing arrays must be non-null");
  const bool isW4 = config().weightDtype == WeightDtype::Mxfp4;
  const bool local = !isW4 && !config().combineRouteWeights && config().worldSize == 1 && config().numExperts == 1 &&
                     config().topK == 1;
  // Cross-input aliases require the original ordered copies rather than concurrent staging.
  const bool fusedStage =
      !local && !overlaps(x, inputBytes, topkIds(), routingBytes) &&
      !overlaps(x, inputBytes, topkWeights(), routingBytes) && !overlaps(ids, routingBytes, input(), inputBytes) &&
      !overlaps(ids, routingBytes, topkWeights(), routingBytes) &&
      !overlaps(scores, routingBytes, input(), inputBytes) && !overlaps(scores, routingBytes, topkIds(), routingBytes);
  if (!isW4) {
    stage(input(), x, inputBytes, stream);
    if (fusedStage) {
      validateStage(topkIds(), ids, routingBytes);
      validateStage(topkWeights(), scores, routingBytes);
    } else {
      stage(topkIds(), ids, routingBytes, stream);
      stage(topkWeights(), scores, routingBytes, stream);
    }
  } else if (fusedStage) {
    validateStage(input(), x, inputBytes);
    validateStage(topkIds(), ids, routingBytes);
    validateStage(topkWeights(), scores, routingBytes);
    if (tokens && (reinterpret_cast<uintptr_t>(x) & 15))
      throw std::invalid_argument("MegaMoE W4A8 direct input must be 16-byte aligned");
  } else {
    stage(input(), x, inputBytes, stream);
    stage(topkIds(), ids, routingBytes, stream);
    stage(topkWeights(), scores, routingBytes, stream);
  }
  auto& r = *impl_->runtime;
  r.startRecorded = false;
  if (signalStart) {
    MSCCLPP_CUDATHROW(cudaMemsetAsync(r.kernelEntrySignal.get(), 0, sizeof(uint32_t), stream));
    MSCCLPP_CUDATHROW(cudaEventRecord(r.startResetEvent, stream));
  }
  auto* signal = signalStart ? r.kernelEntrySignal.get() : nullptr;
  // Fused staging hands the caller arrays to the kernels; ordered copies leave them on the staged workspace.
  const void* launchInput = fusedStage ? x : nullptr;
  const int32_t* launchIds = fusedStage ? ids : nullptr;
  const float* launchScores = fusedStage ? scores : nullptr;
  if (fused) {
    launchNativeW4A8FusedShared(impl_->plan, launchInput, sharedX, launchIds, launchScores, tokens, output,
                                sharedOutput, stream, signal);
  } else if (impl_->jitPlan) {
    char error[ErrorCapacity]{};
    checkJit(impl_->module->launchDirect(impl_->jitPlan->handle, launchInput, launchIds, launchScores, tokens, output,
                                         stream, signal, error, sizeof(error)),
             error, "launch");
  } else if (isW4) {
    launchNativeW4A8(impl_->plan, launchInput, launchIds, launchScores, tokens, output, stream, signal);
  } else {
    launchNativeW8A16(impl_->plan, launchIds, launchScores, tokens, output, stream, signal);
  }
  r.startRecorded = signalStart;
}

void MegaMoeContext::forwardShared(const void* x, void* output, int tokens, cudaStream_t stream) {
  if (config().weightDtype == WeightDtype::Mxfp4 || config().worldSize != 1 || config().numExperts != 1 ||
      config().topK != 1)
    throw std::invalid_argument(
        "MegaMoE forwardShared requires weightDtype=Mxfp8 or Fp8Nv, worldSize=1, numExperts=1, and topK=1");
  const size_t inputBytes = validateForward(x, output, tokens);
  // W8A8 quantizes a 16-byte-aligned caller input directly; W8A16 TMA-loads the staged symmetric input.
  const bool directInput =
      config().activationDtype == ActivationDtype::Fp8Static && (reinterpret_cast<uintptr_t>(x) & 15) == 0;
  if (!directInput) stage(input(), x, inputBytes, stream);
  impl_->runtime->startRecorded = false;
  launchNativeSharedExpert(impl_->plan, tokens, output, stream, directInput ? x : nullptr);
}

void MegaMoeContext::waitUntilStarted(cudaStream_t stream) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE start wait must use the context's CUDA device");
  auto& r = *impl_->runtime;
  if (!r.startRecorded)
    throw std::invalid_argument("MegaMoE waitUntilStarted requires a preceding signalStart forward");
  MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, r.startResetEvent, 0));
  MSCCLPP_CUTHROW(cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(r.kernelEntrySignal.get()), 1,
                                      CU_STREAM_WAIT_VALUE_EQ));
}

}  // namespace mscclpp::megamoe
