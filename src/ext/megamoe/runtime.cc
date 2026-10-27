// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/utils.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe.hpp"
#include "megamoe_jit.hpp"
#include "megamoe_net.hpp"

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
                            int32_t(c.combineRouteWeights),
                            int32_t(c.computeOnly)};
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
  // Absent from modules built before inter-rack support; required only by inter-rack contexts.
  MegaMoeJitCreatePlanNetV1 createPlanNet = nullptr;
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
                              MSCCLPP_MEGAMOE_JIT_ADAPTIVE_CACHE_WEIGHTS))
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
    createPlanNet =
        reinterpret_cast<MegaMoeJitCreatePlanNetV1>(dlsym(handle, MSCCLPP_MEGAMOE_JIT_CREATE_PLAN_NET_ENTRYPOINT));
    (void)dlerror();
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
  std::array<int, 12> config;
  int tag = 0;
  std::array<int, 19> specialization;
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
  info.config = {c.worldSize,
                 c.maxTokens,
                 c.hidden,
                 c.intermediate,
                 c.numExperts,
                 c.topK,
                 c.smMargin,
                 int(c.weightDtype),
                 int(c.activationDtype),
                 int(c.combineRouteWeights),
                 int(c.computeOnly),
                 c.ranksPerRack};
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

// Validate peers and return the ranks per rack. Racks are contiguous rank blocks of one NVLink fabric; other racks are
// reached over IB. Without ranksPerRack, a rack is the NVLink fabric of rank 0's block, or a single rank when the
// fabric is unavailable.
int validatePeers(const std::shared_ptr<Communicator>& comm, const NativeConfig& c, const RankInfo& info) {
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
    for (int previous = 0; previous < rank; ++previous)
      if (!std::strcmp(peer.gpuUuid, peers[previous].gpuUuid))
        throw std::invalid_argument("MegaMoE requires exactly one rank per GPU");
  }
  auto sameFabric = [&](int a, int b) {
    const auto& x = peers[a];
    const auto& y = peers[b];
    return x.fabricReady && y.fabricReady && x.cliqueId == y.cliqueId &&
           !std::memcmp(x.fabricUuid, y.fabricUuid, sizeof(x.fabricUuid));
  };
  int rackSize = c.ranksPerRack;
  const bool detect = !rackSize;
  if (detect) {
    rackSize = 1;
    while (rackSize < c.worldSize && sameFabric(0, rackSize)) ++rackSize;
    if (rackSize > MaxRackSize)
      throw std::invalid_argument("MegaMoE detected an NVLink fabric of more than " + std::to_string(MaxRackSize) +
                                  " ranks; set ranksPerRack");
    if (c.worldSize % rackSize)
      throw std::invalid_argument("MegaMoE detected NVLink racks of unequal size; set ranksPerRack");
  }
  for (int rank = 0; rank < c.worldSize; ++rank) {
    const int leader = rank / rackSize * rackSize;
    if (rank != leader && !sameFabric(leader, rank))
      throw std::invalid_argument(
          "MegaMoE requires the ranks of each rack in one active NVLink fabric; PCIe peers are unsupported");
    // Detected racks must cover whole fabrics, so a fabric never mixes NVLink and IB peers.
    if (detect && rank == leader && rank && sameFabric(rank - 1, rank))
      throw std::invalid_argument("MegaMoE detected NVLink racks of unequal size; set ranksPerRack");
  }
  return rackSize;
}

// The proxy holds the main region, one remote region per network peer, and one result slot per network peer.
static_assert(1 + 2 * (MaxWorldSize - 1) <= (1 << TriggerBitsMemoryId), "port-channel memory IDs are exhausted");
static_assert(MaxWorldSize - 1 <= (1 << TriggerBitsSemaphoreId), "port-channel semaphore IDs are exhausted");

Transport netTransport(int device) {
  static constexpr Transport IBs[] = {Transport::IB0, Transport::IB1, Transport::IB2, Transport::IB3,
                                      Transport::IB4, Transport::IB5, Transport::IB6, Transport::IB7};
  const int count = std::min(getIBDeviceCount(), int(std::size(IBs)));
  if (count <= 0) throw std::runtime_error("MegaMoE inter-rack execution requires an InfiniBand device");
  // As elsewhere in MSCCL++, GPU i uses IB device i; MSCCLPP_HCA_DEVICES remaps the devices.
  return IBs[device % count];
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
  std::shared_ptr<uint32_t> kernelEntrySignal;
  cudaEvent_t startResetEvent = nullptr;
  bool startRecorded = false;
  std::array<std::shared_ptr<uint8_t>, 4> weightBuffers;
  RegisteredMemory localMemory;
  std::vector<RegisteredMemory> peerMemories;
  std::vector<Connection> connections;
  std::shared_ptr<KernelPlan> plan;
  std::shared_ptr<JitModule> module;
  std::unique_ptr<JitPlan> jitPlan;
  // Inter-rack network path; empty when every peer shares the local rack.
  int rackSize = 1;
  NetLayout netLayout;
  std::unique_ptr<GpuBuffer<char>> netBuffer;
  RegisteredMemory netMemory;
  // Owner-side result slots, one registered memory per network source to keep put offsets within 32 bits.
  std::vector<RegisteredMemory> netSlotMemories;
  std::vector<RegisteredMemory> netPeerMemories;
  std::vector<Connection> netConnections;
  std::shared_ptr<ProxyService> proxy;
  bool proxyStarted = false;
  std::shared_ptr<PortChannelDeviceHandle> netChannels;

  ~Impl() {
    if (device < 0) return;
    // Synchronize before destroying plans/imports/allocations; member destruction
    // alone would unmap peer memory before cudaFree waits for pending work.
    int original = -1;
    (void)cudaGetDevice(&original);
    (void)cudaSetDevice(device);
    (void)cudaDeviceSynchronize();
    if (startResetEvent) (void)cudaEventDestroy(startResetEvent);
    // The proxy may still reference the network memories and connections.
    if (proxyStarted) proxy->stopProxy();
    plan.reset();
    jitPlan.reset();
    module.reset();
    proxy.reset();
    netChannels.reset();
    netConnections.clear();
    netPeerMemories.clear();
    netSlotMemories.clear();
    netMemory = RegisteredMemory{};
    netBuffer.reset();
    connections.clear();
    peerMemories.clear();
    localMemory = RegisteredMemory{};
    weightBuffers = {};
    kernelEntrySignal.reset();
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
    if (tag < 0 || tag > std::numeric_limits<int>::max() - 5)
      throw std::invalid_argument("MegaMoE bootstrap tag must be in [0, INT_MAX-5]");
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
    info = rankInfo(c, p.device);
    info.tag = tag;
    info.specialization = {kernelTileM(),
                           kernelTileN(),
                           kernelTileK(),
                           kernelLoadStages(),
                           kernelTransformStages(),
                           kernelNumWarps(),
                           kernelTransferRegisters(),
                           kernelLoadWarps(),
                           int(kernelSplitPipelines()),
                           kernelEpilogueTokens(),
                           kernelEpilogueWarps(),
                           kernelEpilogueRegisters(),
                           kernelDispatchChunk(),
                           kernelDispatchWarps(),
                           kernelDispatchStages(),
                           int(kernelExpertMajorTasks()),
                           int(kernelCacheWeights()),
                           int(kernelAdaptiveCacheWeights()),
                           kernelClusterSize()};
    std::memcpy(info.kernelId, p.kernelId.c_str(), p.kernelId.size() + 1);
    info.symmetricBytes = p.layout.bytes;
    info.privateBytes = p.workspaceBytes;
  });
  p.rackSize = validatePeers(comm, c, info);
  p.netLayout = getNetLayout(c, p.rackSize);
  const int rackBegin = c.rank / p.rackSize * p.rackSize;
  auto isNet = [&](int rank) { return rank < rackBegin || rank >= rackBegin + p.rackSize; };
  p.communicator = comm;
  PackedWeights packed{};
  collectiveCheck(comm, "allocation and weight packing", [&] {
    p.symmetric = std::make_unique<GpuBuffer<char>>(p.layout.bytes);
    p.workspace = mscclpp::detail::gpuCallocShared<char>(p.workspaceBytes);
    p.peerBases = mscclpp::detail::gpuCallocShared<uint64_t>(c.worldSize);
    p.kernelEntrySignal = mscclpp::detail::gpuCallocShared<uint32_t>(1);
    MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&p.startResetEvent, cudaEventDisableTiming));
    p.localMemory = p.communicator->registerMemory(p.symmetric->data(), p.layout.bytes, Transport::CudaIpc);
    if (p.netLayout.bytes) {
      p.netBuffer = std::make_unique<GpuBuffer<char>>(p.netLayout.bytes);
      const Transport transport = netTransport(p.device);
      p.netMemory = p.communicator->registerMemory(p.netBuffer->data(), p.netLayout.mainBytes, transport);
      for (int index = 0; index < c.worldSize - p.rackSize; ++index)
        p.netSlotMemories.push_back(p.communicator->registerMemory(
            p.netBuffer->data() + p.netLayout.combineOut + size_t(index) * p.netLayout.slotBytes, p.netLayout.slotBytes,
            transport));
    }
    size_t localExperts = c.numExperts / c.worldSize;
    const size_t valuesPerByte = c.weightDtype == WeightDtype::Mxfp4 ? 2 : 1;
    const std::array<size_t, 4> sizes{
        product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden)}) / valuesPerByte,
        product({localExperts, 2, size_t(c.intermediate), size_t(c.hidden / 32)}),
        product({localExperts, size_t(c.hidden), size_t(c.intermediate)}) / valuesPerByte,
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
  std::vector<std::shared_future<Semaphore>> semaphoreFutures;
  std::vector<uint64_t> peerPointers;
  collectiveCheck(comm, "peer exchange setup", [&] {
    p.peerMemories.resize(c.worldSize);
    p.peerMemories[c.rank] = p.localMemory;
    p.netPeerMemories.resize(c.worldSize);
    memoryFutures.resize(c.worldSize);
    connectionFutures.resize(c.worldSize);
    peerPointers.resize(c.worldSize);
    for (int rank = 0; rank < c.worldSize; ++rank) {
      if (rank == c.rank) continue;
      if (isNet(rank)) {
        EndpointConfig endpoint;
        endpoint.transport = netTransport(p.device);
        connectionFutures[rank] = p.communicator->connect(endpoint, rank, tag + 2);
        p.communicator->sendMemory(p.netMemory, rank, tag + 3);
        memoryFutures[rank] = p.communicator->recvMemory(rank, tag + 3);
        continue;
      }
      connectionFutures[rank] = p.communicator->connect(Transport::CudaIpc, rank, tag);
      p.communicator->sendMemory(p.localMemory, rank, tag + 1);
      memoryFutures[rank] = p.communicator->recvMemory(rank, tag + 1);
    }
  });
  collectiveCheck(comm, "peer mapping", [&] {
    semaphoreFutures.resize(c.worldSize);
    for (int rank = 0; rank < c.worldSize; ++rank) {
      if (isNet(rank)) {
        p.netConnections.push_back(connectionFutures[rank].get());
        p.netPeerMemories[rank] = memoryFutures[rank].get();
        if (p.netPeerMemories[rank].size() != p.netLayout.mainBytes)
          throw std::runtime_error("MegaMoE could not register a matching network peer workspace");
        // One semaphore per connection: IB modes without RDMA atomics forward a single signal address per connection.
        semaphoreFutures[rank] = p.communicator->buildSemaphore(p.netConnections.back(), rank, tag + 4);
        peerPointers[rank] = 0;
        continue;
      }
      if (rank != c.rank) {
        p.connections.push_back(connectionFutures[rank].get());
        p.peerMemories[rank] = memoryFutures[rank].get();
      }
      if (!p.peerMemories[rank].data() || p.peerMemories[rank].size() != p.layout.bytes)
        throw std::runtime_error("MegaMoE could not map a matching CudaIpc peer workspace");
      peerPointers[rank] = reinterpret_cast<uint64_t>(p.peerMemories[rank].data());
    }
  });
  if (p.netLayout.bytes) {
    collectiveCheck(comm, "network channel setup", [&] {
      p.proxy = std::make_shared<ProxyService>();
      const MemoryId local = p.proxy->addMemory(p.netMemory);
      std::vector<PortChannelDeviceHandle> channels(2 * c.worldSize);
      for (int rank = 0; rank < c.worldSize; ++rank) {
        if (!isNet(rank)) continue;
        const MemoryId remote = p.proxy->addMemory(p.netPeerMemories[rank]);
        const MemoryId slot = p.proxy->addMemory(p.netSlotMemories[rank < rackBegin ? rank : rank - p.rackSize]);
        const SemaphoreId semaphore = p.proxy->addSemaphore(semaphoreFutures[rank].get());
        channels[rank] = p.proxy->portChannel(semaphore, remote, local).deviceHandle();
        channels[c.worldSize + rank] = p.proxy->portChannel(semaphore, remote, slot).deviceHandle();
      }
      p.netChannels = mscclpp::detail::gpuCallocShared<PortChannelDeviceHandle>(channels.size());
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(p.netChannels.get(), channels.data(),
                                        channels.size() * sizeof(PortChannelDeviceHandle), cudaMemcpyHostToDevice,
                                        stream));
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      p.proxy->startProxy();
      p.proxyStarted = true;
    });
  }
  collectiveCheck(comm, "plan creation", [&] {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(p.peerBases.get(), peerPointers.data(), c.worldSize * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    const NetPlan net{p.rackSize, p.netBuffer ? p.netBuffer->data() : nullptr, p.netChannels.get()};
    if (p.module) {
      p.jitPlan = std::make_unique<JitPlan>(p.module);
      const auto config = jitConfig(c);
      const auto weights = jitWeights(packed);
      char error[ErrorCapacity]{};
      if (p.module->createPlanNet) {
        const MegaMoeJitNetV1 jitNet{net.rackSize, 0, p.netLayout.bytes, net.buffer, net.channels};
        checkJit(p.module->createPlanNet(&config, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), &weights,
                                         &jitNet, &p.jitPlan->handle, error, sizeof(error)),
                 error, "plan creation");
      } else if (p.netLayout.bytes) {
        throw std::invalid_argument("MegaMoE JIT module predates inter-rack support; recompile it");
      } else {
        checkJit(p.module->api->createPlan(&config, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), &weights,
                                           &p.jitPlan->handle, error, sizeof(error)),
                 error, "plan creation");
      }
      if (!p.jitPlan->handle) throw std::runtime_error("MegaMoE JIT module returned a null plan");
    } else {
      p.plan = createKernelPlan(c, p.symmetric->data(), p.peerBases.get(), p.workspace.get(), packed, &net);
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
int MegaMoeContext::ranksPerRack() const { return impl_->rackSize; }
size_t MegaMoeContext::networkBytes() const { return impl_->netLayout.bytes; }
const std::string& MegaMoeContext::kernelId() const { return impl_->kernelId; }
int MegaMoeContext::kernelTileM() const {
  if (impl_->module) return impl_->module->api->tileM;
  return detail::W4TileM;
}
int MegaMoeContext::kernelClusterSize() const {
  if (impl_->module) return impl_->module->api->clusterSize;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4ClusterSize : 2;
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
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4LoadStages : detail::W8LoadStages;
}
int MegaMoeContext::kernelTransformStages() const {
  return config().weightDtype == WeightDtype::Mxfp4 ? 0 : detail::W8TransformStages;
}
int MegaMoeContext::kernelNumWarps() const {
  if (impl_->module) return impl_->module->api->numWarps;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4NumWarps : detail::W8A16WarpSchedule<false>::NumWarps;
}
int MegaMoeContext::kernelTransferRegisters() const {
  if (impl_->module) return impl_->module->api->transferRegisters;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4TransferRegisters : 32;
}
int MegaMoeContext::kernelLoadWarps() const {
  if (impl_->module) return impl_->module->api->loadWarps;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4LoadWarps : 2;
}
bool MegaMoeContext::kernelSplitPipelines() const {
  if (impl_->module) return bool(impl_->module->api->splitPipelines);
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4SplitPipelines : false;
}
int MegaMoeContext::kernelEpilogueTokens() const {
  if (impl_->module) return impl_->module->api->epilogueTokens;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4EpilogueTokens : 32;
}
int MegaMoeContext::kernelEpilogueWarps() const {
  if (impl_->module) return impl_->module->api->epilogueWarps;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4EpilogueWarps : 4;
}
int MegaMoeContext::kernelEpilogueRegisters() const {
  if (impl_->module) return impl_->module->api->epilogueRegisters;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4EpilogueRegisters : 224;
}
int MegaMoeContext::kernelDispatchChunk() const {
  if (impl_->module) return impl_->module->api->dispatchChunk;
  return detail::W4DispatchChunk;
}
int MegaMoeContext::kernelDispatchWarps() const {
  if (impl_->module) return impl_->module->api->dispatchWarps;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4DispatchWarps : 4;
}
int MegaMoeContext::kernelDispatchStages() const {
  if (impl_->module) return impl_->module->api->dispatchStages;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4DispatchStages : 2;
}
bool MegaMoeContext::kernelFixedTokenCount() const {
  if (impl_->module) return impl_->module->fixedTokenCount;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4FixedTokenCount : false;
}
bool MegaMoeContext::kernelExpertMajorTasks() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_EXPERT_MAJOR_TASKS) != 0;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4ExpertMajorTasks : false;
}
bool MegaMoeContext::kernelCacheWeights() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_CACHE_WEIGHTS) != 0;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4CacheWeights : false;
}
bool MegaMoeContext::kernelAdaptiveCacheWeights() const {
  if (impl_->module) return (impl_->module->optimizationFlags & MSCCLPP_MEGAMOE_JIT_ADAPTIVE_CACHE_WEIGHTS) != 0;
  return config().weightDtype == WeightDtype::Mxfp4 ? detail::W4AdaptiveCacheWeights : false;
}

size_t MegaMoeContext::validateForward(const void* x, void* output, int tokens) const {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE forward must use the context's CUDA device");
  if (tokens < 0 || tokens > config().maxTokens)
    throw std::invalid_argument("MegaMoE numTokens exceeds the configured capacity");
  size_t inputBytes = size_t(tokens) * config().hidden * 2;
  size_t outputBytes =
      config().computeOnly ? size_t(config().worldSize) * tokens * config().topK * config().hidden * 2 : inputBytes;
  if ((tokens && (!x || !output)) || overlaps(output, outputBytes, impl_->symmetric->data(), impl_->layout.bytes))
    throw std::invalid_argument("MegaMoE requires non-null arrays and output disjoint from the registered workspace");
  return inputBytes;
}

void MegaMoeContext::forward(const void* x, const int32_t* ids, const float* scores, void* output, int tokens,
                             cudaStream_t stream, bool signalStart, const uint8_t* inputScales) {
  const size_t inputBytes = validateForward(x, output, tokens);
  const size_t routingBytes = size_t(tokens) * config().topK * 4;
  if (tokens && (!ids || (!config().combineRouteWeights && !scores)))
    throw std::invalid_argument("MegaMoE routing arrays must be non-null");
  if (config().computeOnly && tokens && config().weightDtype == WeightDtype::Mxfp4 &&
      config().activationDtype == ActivationDtype::Mxfp8 && !inputScales)
    throw std::invalid_argument("MegaMoE computeOnly W4A8 MXFP8 requires inputScales");
  const bool isW4 = config().weightDtype == WeightDtype::Mxfp4;
  const bool local = !isW4 && !config().combineRouteWeights && config().worldSize == 1 && config().numExperts == 1 &&
                     config().topK == 1;
  // Cross-input aliases require the original ordered copies rather than concurrent staging.
  const bool fusedStage =
      !local && !config().computeOnly && !overlaps(x, inputBytes, topkIds(), routingBytes) &&
      !overlaps(x, inputBytes, topkWeights(), routingBytes) && !overlaps(ids, routingBytes, input(), inputBytes) &&
      !overlaps(ids, routingBytes, topkWeights(), routingBytes) &&
      !overlaps(scores, routingBytes, input(), inputBytes) && !overlaps(scores, routingBytes, topkIds(), routingBytes);
  if (!config().computeOnly) {
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
  }
  impl_->startRecorded = false;
  if (signalStart) {
    MSCCLPP_CUDATHROW(cudaMemsetAsync(impl_->kernelEntrySignal.get(), 0, sizeof(uint32_t), stream));
    MSCCLPP_CUDATHROW(cudaEventRecord(impl_->startResetEvent, stream));
  }
  auto* signal = signalStart ? impl_->kernelEntrySignal.get() : nullptr;
  // Fused staging and computeOnly hand the caller arrays directly to the kernels.
  const bool directLaunch = fusedStage || config().computeOnly;
  const void* launchInput = directLaunch ? x : nullptr;
  const int32_t* launchIds = directLaunch ? ids : nullptr;
  const float* launchScores = directLaunch ? scores : nullptr;
  if (impl_->jitPlan) {
    char error[ErrorCapacity]{};
    checkJit(impl_->module->launchDirect(impl_->jitPlan->handle, launchInput, inputScales, launchIds, launchScores,
                                         tokens, output, stream, signal, error, sizeof(error)),
             error, "launch");
  } else if (isW4) {
    launchNativeW4A8(impl_->plan, launchInput, launchIds, launchScores, tokens, output, stream, signal, inputScales);
  } else {
    launchNativeW8A16(impl_->plan, launchIds, launchScores, tokens, output, stream, signal, launchInput);
  }
  impl_->startRecorded = signalStart;
}

void MegaMoeContext::forwardShared(const void* x, void* output, int tokens, cudaStream_t stream) {
  if (config().weightDtype != WeightDtype::Mxfp8 || config().worldSize != 1 || config().numExperts != 1 ||
      config().topK != 1)
    throw std::invalid_argument(
        "MegaMoE forwardShared requires weightDtype=Mxfp8, worldSize=1, numExperts=1, and topK=1");
  const size_t inputBytes = validateForward(x, output, tokens);
  stage(input(), x, inputBytes, stream);
  impl_->startRecorded = false;
  launchNativeSharedExpert(impl_->plan, tokens, output, stream);
}

void MegaMoeContext::waitUntilStarted(cudaStream_t stream) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != impl_->device) throw std::invalid_argument("MegaMoE start wait must use the context's CUDA device");
  if (!impl_->startRecorded)
    throw std::invalid_argument("MegaMoE waitUntilStarted requires a preceding signalStart forward");
  MSCCLPP_CUDATHROW(cudaStreamWaitEvent(stream, impl_->startResetEvent, 0));
  MSCCLPP_CUTHROW(cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(impl_->kernelEntrySignal.get()), 1,
                                      CU_STREAM_WAIT_VALUE_EQ));
}

}  // namespace mscclpp::megamoe
