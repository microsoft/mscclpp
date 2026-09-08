// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "host/gpu_net_io_service.hpp"

#include <cuda_runtime.h>
#include <dirent.h>
#include <endian.h>
#include <infiniband/verbs.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mscclpp/errors.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ib.hpp"  // mscclpp core IbCtx / IbMr (ibverbs context + pd + MR)

// Vendored DOCA GPUNetIO host API.
extern "C" {
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_high_level.h"
#include "host/doca_verbs.h"
}

namespace mscclpp {

namespace {

#define MSCCLPP_DOCA_THROW(expr)                                                                                   \
  do {                                                                                                             \
    doca_error_t _st = (expr);                                                                                     \
    if (_st != DOCA_SUCCESS) {                                                                                     \
      throw mscclpp::Error(                                                                                        \
          std::string("DOCA GPUNetIO call failed: ") + #expr + " status=" + std::to_string(static_cast<int>(_st)), \
          mscclpp::ErrorCode::SystemError);                                                                        \
    }                                                                                                              \
  } while (0)

#define MSCCLPP_CUDA_THROW(expr)                                                                       \
  do {                                                                                                 \
    cudaError_t _e = (expr);                                                                           \
    if (_e != cudaSuccess) {                                                                           \
      throw mscclpp::Error(std::string("CUDA call failed: ") + #expr + " : " + cudaGetErrorString(_e), \
                           mscclpp::ErrorCode::SystemError);                                           \
    }                                                                                                  \
  } while (0)

// Per-peer connection info exchanged via the bootstrap all-gather.
struct QpExchangeInfo {
  uint32_t qpn;
  uint16_t lid;
  uint16_t gidIndex;
  uint8_t gid[16];
  uint8_t linkLayer;
  uint8_t grhRequired;
  uint8_t activeMtu;
  uint8_t pad;
};

// Per-rank memory info exchanged via the bootstrap all-gather.
struct MemExchangeInfo {
  uint64_t base;  // symmetric-buffer base address on that rank
  uint32_t rkey;  // remote key for that rank's registered MR
  uint32_t pad;
};

struct ConfigExchangeInfo {
  uint32_t numHcas;
  uint32_t numQpsPerPeer;
};

struct TopologyExchangeInfo {
  uint64_t hostHash;
  int32_t gpuNumaNode;
  char gpuPciBusId[32];
};

struct HcaTopology {
  std::string name;
  std::string pciPath;
  int numaNode;
};

uint64_t stableStringHash(const std::string& value) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char ch : value) {
    hash ^= ch;
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t localHostHash() {
  char hostName[HOST_NAME_MAX + 1] = {0};
  if (gethostname(hostName, HOST_NAME_MAX) != 0) {
    throw Error("gethostname failed during GPUNetIO topology discovery", ErrorCode::SystemError);
  }
  return stableStringHash(hostName);
}

int readNumaNode(const std::string& path) {
  std::ifstream file(path + "/numa_node");
  int numaNode = -1;
  if (file.is_open()) file >> numaNode;
  return numaNode;
}

int readPortState(const std::string& path) {
  std::ifstream file(path + "/ports/1/state");
  int state = 0;
  if (file.is_open()) file >> state;
  return state;
}

std::string canonicalPath(const std::string& path) {
  char resolved[PATH_MAX] = {0};
  return realpath(path.c_str(), resolved) == nullptr ? std::string() : std::string(resolved);
}

int pciPathDistance(const std::string& left, const std::string& right) {
  if (left.empty() || right.empty()) return 1 << 10;
  size_t common = 0;
  size_t leftPos = 0;
  size_t rightPos = 0;
  int leftDepth = 0;
  int rightDepth = 0;
  while (leftPos < left.size() && rightPos < right.size()) {
    const size_t leftEnd = left.find('/', leftPos + 1);
    const size_t rightEnd = right.find('/', rightPos + 1);
    const size_t leftLen = (leftEnd == std::string::npos ? left.size() : leftEnd) - leftPos;
    const size_t rightLen = (rightEnd == std::string::npos ? right.size() : rightEnd) - rightPos;
    if (leftLen != rightLen || left.compare(leftPos, leftLen, right, rightPos, rightLen) != 0) break;
    common++;
    leftPos = leftEnd == std::string::npos ? left.size() : leftEnd;
    rightPos = rightEnd == std::string::npos ? right.size() : rightEnd;
  }
  leftDepth = static_cast<int>(std::count(left.begin(), left.end(), '/'));
  rightDepth = static_cast<int>(std::count(right.begin(), right.end(), '/'));
  return leftDepth + rightDepth - 2 * static_cast<int>(common);
}

std::vector<HcaTopology> discoverActiveHcas() {
  constexpr const char* InfinibandClassPath = "/sys/class/infiniband";
  std::unique_ptr<DIR, decltype(&closedir)> directory(opendir(InfinibandClassPath), &closedir);
  if (directory == nullptr) {
    throw Error("Failed to open /sys/class/infiniband during GPUNetIO topology discovery", ErrorCode::SystemError);
  }

  std::vector<HcaTopology> hcas;
  while (struct dirent* entry = readdir(directory.get())) {
    if (entry->d_name[0] == '.') continue;
    const std::string name = entry->d_name;
    const std::string classPath = std::string(InfinibandClassPath) + "/" + name;
    const std::string devicePath = classPath + "/device";
    const std::string driverPath = canonicalPath(devicePath + "/driver");
    if (driverPath.find("/mlx5_core") == std::string::npos) continue;
    if (readPortState(classPath) != IBV_PORT_ACTIVE) continue;

    hcas.push_back({name, canonicalPath(devicePath), readNumaNode(devicePath)});
  }
  std::sort(hcas.begin(), hcas.end(),
            [](const HcaTopology& left, const HcaTopology& right) { return left.name < right.name; });
  if (hcas.empty()) {
    throw Error("GPUNetIO automatic topology discovery found no active port-1 IB devices", ErrorCode::InvalidUsage);
  }
  return hcas;
}

int hcaAffinityScore(const TopologyExchangeInfo& gpu, const HcaTopology& hca) {
  constexpr int NumaMismatchPenalty = 1 << 20;
  const std::string gpuPath = canonicalPath("/sys/bus/pci/devices/" + std::string(gpu.gpuPciBusId));
  int score = pciPathDistance(gpuPath, hca.pciPath);
  if (gpu.gpuNumaNode >= 0 && hca.numaNode >= 0 && gpu.gpuNumaNode != hca.numaNode) {
    score += NumaMismatchPenalty;
  }
  return score;
}

std::vector<std::string> selectAutomaticHcas(const std::vector<TopologyExchangeInfo>& topology, int rank) {
  const auto hcas = discoverActiveHcas();
  std::vector<int> localRanks;
  for (int peer = 0; peer < static_cast<int>(topology.size()); ++peer) {
    if (topology[peer].hostHash == topology[rank].hostHash) localRanks.push_back(peer);
  }
  if (localRanks.empty()) {
    throw Error("GPUNetIO topology discovery could not find the local rank", ErrorCode::InternalError);
  }

  const int hcaCount = static_cast<int>(hcas.size());
  const int localRankCount = static_cast<int>(localRanks.size());
  std::vector<std::vector<int>> affinity(localRankCount, std::vector<int>(hcaCount));
  for (int localRank = 0; localRank < localRankCount; ++localRank) {
    for (int hca = 0; hca < hcaCount; ++hca) {
      affinity[localRank][hca] = hcaAffinityScore(topology[localRanks[localRank]], hcas[hca]);
    }
  }

  std::vector<std::vector<int>> localHcas(localRankCount);
  std::vector<int> hcasPerRank(localRankCount);
  int maxHcasPerRank = 0;
  for (int localRank = 0; localRank < localRankCount; ++localRank) {
    const int bestAffinity = *std::min_element(affinity[localRank].begin(), affinity[localRank].end());
    for (int hca = 0; hca < hcaCount; ++hca) {
      if (affinity[localRank][hca] == bestAffinity) localHcas[localRank].push_back(hca);
    }

    const int firstHca = localHcas[localRank][0];
    int bestGpuAffinity = std::numeric_limits<int>::max();
    for (int gpu = 0; gpu < localRankCount; ++gpu) {
      bestGpuAffinity = std::min(bestGpuAffinity, affinity[gpu][firstHca]);
    }
    int localGpuCount = 0;
    for (int gpu = 0; gpu < localRankCount; ++gpu) {
      if (affinity[gpu][firstHca] == bestGpuAffinity) localGpuCount++;
    }
    hcasPerRank[localRank] = (static_cast<int>(localHcas[localRank].size()) + localGpuCount - 1) / localGpuCount;
    maxHcasPerRank = std::max(maxHcasPerRank, hcasPerRank[localRank]);
  }

  std::vector<std::vector<int>> assignments(localRanks.size());
  std::vector<int> usage(hcaCount, 0);
  for (int slot = 0; slot < maxHcasPerRank; ++slot) {
    for (int localRank = 0; localRank < localRankCount; ++localRank) {
      if (slot >= hcasPerRank[localRank]) continue;
      int selected = -1;
      int selectedUsage = std::numeric_limits<int>::max();
      int selectedOrder = std::numeric_limits<int>::max();
      const int candidateCount = static_cast<int>(localHcas[localRank].size());
      for (int candidate = 0; candidate < candidateCount; ++candidate) {
        const int order = (candidate - localRank % candidateCount + candidateCount) % candidateCount;
        const int hca = localHcas[localRank][candidate];
        if (std::find(assignments[localRank].begin(), assignments[localRank].end(), hca) !=
            assignments[localRank].end()) {
          continue;
        }
        if (usage[hca] < selectedUsage || (usage[hca] == selectedUsage && order < selectedOrder)) {
          selected = hca;
          selectedUsage = usage[hca];
          selectedOrder = order;
        }
      }
      if (selected >= 0) {
        assignments[localRank].push_back(selected);
        usage[selected]++;
      }
    }
  }

  const auto rankIt = std::find(localRanks.begin(), localRanks.end(), rank);
  const auto& selected = assignments[std::distance(localRanks.begin(), rankIt)];
  std::vector<std::string> names;
  names.reserve(selected.size());
  for (int hca : selected) names.push_back(hcas[hca].name);
  return names;
}

std::vector<std::string> splitIbDeviceNames(const std::string& spec) {
  std::vector<std::string> names;
  size_t begin = 0;
  while (begin <= spec.size()) {
    const size_t end = spec.find(',', begin);
    const size_t count = end == std::string::npos ? spec.size() - begin : end - begin;
    const size_t first = spec.find_first_not_of(" \t", begin);
    const size_t limit = begin + count;
    if (first != std::string::npos && first < limit) {
      const size_t last = spec.find_last_not_of(" \t", limit - 1);
      names.emplace_back(spec.substr(first, last - first + 1));
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (names.empty()) throw Error("GPUNetIO HCA list is empty", ErrorCode::InvalidUsage);
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (names[i] == names[j]) throw Error("GPUNetIO HCA list contains a duplicate", ErrorCode::InvalidUsage);
    }
  }
  return names;
}

}  // namespace

struct GpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  std::vector<std::string> ibDeviceNames;
  bool automaticHcaSelection = false;
  int cudaDeviceId;
  int rank = -1;
  int worldSize = 0;
  bool didSetup = false;

  struct HcaContext {
    std::string deviceName;
    std::unique_ptr<IbCtx> ibCtx;
    std::unique_ptr<const IbMr> mr;
  };
  std::vector<HcaContext> hcas;

  struct doca_gpu* gpuDev = nullptr;
  int portNum = 1;
  int gidIndex = 0;
  int numQpsPerPeer = 1;  // MSCCLPP_EP_GPUNETIO_QPS_PER_PEER (>= 1)

  // One high-level QP per (remote rank, logical QP); self entries are null.
  std::vector<struct doca_gpu_verbs_qp_hl*> qpHl;
  doca_gpu_verbs_service_t cpuProxyService = nullptr;
  struct doca_gpu_dev_verbs_qp* qpFlatGpu = nullptr;  // GPU array from flat_list

  // Device-side arrays referenced by GpuNetIoDeviceContext.
  uint32_t* rkeysGpu = nullptr;
  uint32_t* lkeysGpu = nullptr;
  uintptr_t* peerBaseGpu = nullptr;
  GpuNetIoDeviceContext* ctxGpu = nullptr;

  ~Impl() {
    if (cpuProxyService) (void)doca_gpu_verbs_destroy_service(cpuProxyService);
    if (ctxGpu) (void)cudaFree(ctxGpu);
    if (rkeysGpu) (void)cudaFree(rkeysGpu);
    if (lkeysGpu) (void)cudaFree(lkeysGpu);
    if (peerBaseGpu) (void)cudaFree(peerBaseGpu);
    if (qpFlatGpu) (void)doca_gpu_verbs_qp_flat_list_destroy_hl(qpFlatGpu);
    for (auto* q : qpHl) {
      if (q) (void)doca_gpu_verbs_destroy_qp_hl(q);
    }
    if (gpuDev) (void)doca_gpu_destroy(gpuDev);
  }

  int numHcas() const { return static_cast<int>(hcas.size()); }
  int hcaIndex(int qpIndex) const { return qpIndex % numHcas(); }
  size_t qpIndex(int peer, int qp) const { return static_cast<size_t>(peer) * numQpsPerPeer + qp; }

  void queryLocalPort(int hcaIndex, QpExchangeInfo& info) {
    auto& hca = hcas[hcaIndex];
    struct ibv_port_attr portAttr;
    std::memset(&portAttr, 0, sizeof(portAttr));
    if (ibv_query_port(hca.ibCtx->getContext(), portNum, &portAttr) != 0) {
      throw Error("ibv_query_port failed for GPUNetIO service", ErrorCode::SystemError);
    }
    info.lid = portAttr.lid;
    info.linkLayer = portAttr.link_layer;
    info.grhRequired = (portAttr.flags & IBV_QPF_GRH_REQUIRED) ? 1 : 0;
    info.activeMtu = portAttr.active_mtu;
  }

  void queryLocalGid(int hcaIndex, uint8_t outGid[16]) {
    auto& hca = hcas[hcaIndex];
    union ibv_gid gid;
    std::memset(&gid, 0, sizeof(gid));
    if (ibv_query_gid(hca.ibCtx->getContext(), portNum, gidIndex, &gid) != 0) {
      throw Error("ibv_query_gid failed for GPUNetIO service", ErrorCode::SystemError);
    }
    std::memcpy(outGid, gid.raw, 16);
  }

  doca_verbs_mtu_size pathMtu(const QpExchangeInfo& remote) const {
    uint8_t mtu = remote.activeMtu;
    if (mtu == 0) {
      mtu = IBV_MTU_1024;
    }
    switch (mtu) {
      case IBV_MTU_256:
        return DOCA_VERBS_MTU_SIZE_256_BYTES;
      case IBV_MTU_512:
        return DOCA_VERBS_MTU_SIZE_512_BYTES;
      case IBV_MTU_1024:
        return DOCA_VERBS_MTU_SIZE_1K_BYTES;
      case IBV_MTU_2048:
        return DOCA_VERBS_MTU_SIZE_2K_BYTES;
      case IBV_MTU_4096:
        return DOCA_VERBS_MTU_SIZE_4K_BYTES;
      default:
        return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    }
  }

  // INIT -> RTR -> RTS for one QP, targeting the given remote info.
  void connectQp(int hcaIndex, struct doca_gpu_verbs_qp_hl* qp, const QpExchangeInfo& remote) {
    auto& hca = hcas[hcaIndex];
    struct doca_verbs_ah_attr* ah = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_create(hca.ibCtx->getContext(), &ah));
    struct doca_verbs_gid vgid;
    std::memcpy(vgid.raw, remote.gid, 16);
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_gid(ah, vgid));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_dlid(ah, remote.lid));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sl(ah, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sgid_index(ah, gidIndex));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_hop_limit(ah, 255));
    if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND && !remote.grhRequired) {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
    } else if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND) {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_GRH));
    } else {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IPv6));
    }

    struct doca_verbs_qp_attr* attr = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_create(&attr));

    // RST -> INIT
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_port_num(attr, portNum));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_atomic(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                                DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                                                DOCA_VERBS_QP_ATTR_PORT_NUM));

    // INIT -> RTR
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rq_psn(attr, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_dest_qp_num(attr, remote.qpn));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_path_mtu(attr, pathMtu(remote)));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ah_attr(attr, ah));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_min_rnr_timer(attr, 12));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                                DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                                DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                                DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC));

    // RTR -> RTS
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_sq_psn(attr, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ack_timeout(attr, 18));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_retry_cnt(attr, 7));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rnr_retry(attr, 7));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_rd_atomic(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                                DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                                DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));

    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_destroy(attr));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_destroy(ah));
  }
};

GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceNames,
                                 int cudaDeviceId)
    : pimpl_(std::make_unique<Impl>()) {
  pimpl_->bootstrap = bootstrap;
  pimpl_->automaticHcaSelection = ibDeviceNames.empty();
  if (!pimpl_->automaticHcaSelection) pimpl_->ibDeviceNames = splitIbDeviceNames(ibDeviceNames);
  pimpl_->cudaDeviceId = cudaDeviceId;
  pimpl_->rank = bootstrap->getRank();
  pimpl_->worldSize = bootstrap->getNranks();
}

GpuNetIoService::~GpuNetIoService() = default;

void GpuNetIoService::setup(void* symmetricBuffer, size_t bytes) {
  auto& s = *pimpl_;
  if (s.didSetup) {
    throw Error("GpuNetIoService::setup called more than once", ErrorCode::InvalidUsage);
  }
  s.didSetup = true;

  TopologyExchangeInfo localTopology{};
  localTopology.hostHash = localHostHash();
  MSCCLPP_CUDA_THROW(
      cudaDeviceGetPCIBusId(localTopology.gpuPciBusId, sizeof(localTopology.gpuPciBusId), s.cudaDeviceId));
  localTopology.gpuNumaNode = readNumaNode("/sys/bus/pci/devices/" + std::string(localTopology.gpuPciBusId));
  std::vector<TopologyExchangeInfo> topologyAll(s.worldSize);
  topologyAll[s.rank] = localTopology;
  s.bootstrap->allGather(topologyAll.data(), static_cast<int>(sizeof(TopologyExchangeInfo)));
  if (s.automaticHcaSelection) s.ibDeviceNames = selectAutomaticHcas(topologyAll, s.rank);

  // 1. Resolve the total logical QP count. Multi-HCA mode stripes q over HCAs,
  //    so use at least one QP per HCA and require an even distribution.
  const char* qpsEnv = std::getenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER");
  int nQp = qpsEnv == nullptr ? static_cast<int>(s.ibDeviceNames.size()) : std::atoi(qpsEnv);
  if (nQp < 1) nQp = 1;
  const int nHcas = static_cast<int>(s.ibDeviceNames.size());
  if (nQp < nHcas || nQp % nHcas != 0) {
    throw Error("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER must be a positive multiple of the HCA count",
                ErrorCode::InvalidUsage);
  }
  s.numQpsPerPeer = nQp;

  // All ranks must agree before variable-sized QP/key all-gathers begin.
  std::vector<ConfigExchangeInfo> configAll(s.worldSize);
  configAll[s.rank] = {static_cast<uint32_t>(nHcas), static_cast<uint32_t>(nQp)};
  s.bootstrap->allGather(configAll.data(), static_cast<int>(sizeof(ConfigExchangeInfo)));
  for (const auto& config : configAll) {
    if (config.numHcas != static_cast<uint32_t>(nHcas) || config.numQpsPerPeer != static_cast<uint32_t>(nQp)) {
      throw Error("GPUNetIO ranks disagree on HCA or QP count", ErrorCode::InvalidUsage);
    }
  }

  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO") != nullptr) {
    std::string devices;
    for (int hca = 0; hca < nHcas; ++hca) {
      if (hca != 0) devices += ',';
      devices += s.ibDeviceNames[hca];
    }
    std::fprintf(stderr,
                 "[EPGPUNETIO] rank=%d hcaSelection=%s gpu=%s numa=%d hcas=%s numHcas=%d qpsPerPeer=%d "
                 "qpsPerHca=%d\n",
                 s.rank, s.automaticHcaSelection ? "auto" : "explicit", localTopology.gpuPciBusId,
                 localTopology.gpuNumaNode, devices.c_str(), nHcas, nQp, nQp / nHcas);
    std::fflush(stderr);
  }

  // 2. Register the same symmetric buffer independently on every HCA.
  s.hcas.reserve(nHcas);
  for (const auto& deviceName : s.ibDeviceNames) {
    Impl::HcaContext hca;
    hca.deviceName = deviceName;
    hca.ibCtx = std::make_unique<IbCtx>(deviceName);
    hca.mr = hca.ibCtx->registerMr(symmetricBuffer, bytes);
    s.hcas.emplace_back(std::move(hca));
  }

  // 3. One DOCA GPU handle is shared by QPs from all HCAs.
  char pciBusId[32] = {0};
  MSCCLPP_CUDA_THROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), s.cudaDeviceId));
  MSCCLPP_DOCA_THROW(doca_gpu_create(pciBusId, &s.gpuDev));

  // 4. Create total logical QPs per peer, assigning q to HCA q%nHcas.
  s.qpHl.assign(static_cast<size_t>(s.worldSize) * nQp, nullptr);
  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int q = 0; q < nQp; ++q) {
      const int hcaIndex = s.hcaIndex(q);
      struct doca_gpu_verbs_qp_init_attr_hl initAttr;
      std::memset(&initAttr, 0, sizeof(initAttr));
      initAttr.gpu_dev = s.gpuDev;
      initAttr.ibpd = s.hcas[hcaIndex].ibCtx->getPd();
      initAttr.sq_nwqe = 1024;
      initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
      initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
      MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_qp_hl(&initAttr, &s.qpHl[s.qpIndex(r, q)]));
    }
  }

  // 5. Exchange QP info (all-gather) and connect matching HCA/QP indices.
  const size_t rowLen = static_cast<size_t>(s.worldSize) * nQp;  // QPs this rank publishes
  std::vector<QpExchangeInfo> qpInfo(rowLen);
  std::memset(qpInfo.data(), 0, qpInfo.size() * sizeof(QpExchangeInfo));
  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int q = 0; q < nQp; ++q) {
      const int hcaIndex = s.hcaIndex(q);
      QpExchangeInfo& info = qpInfo[s.qpIndex(r, q)];
      info.qpn = doca_verbs_qp_get_qpn(s.qpHl[s.qpIndex(r, q)]->qp);
      info.gidIndex = static_cast<uint16_t>(s.gidIndex);
      s.queryLocalPort(hcaIndex, info);
      s.queryLocalGid(hcaIndex, info.gid);
    }
  }
  // Each rank publishes, for every peer, its nQp QPs targeting that peer. The
  // all-gather delivers a [worldSize][worldSize*nQp] table; entry [src][dst*nQp+k]
  // is src's k-th QP that talks to dst. We read [r][rank*nQp+k] to pair k<->k.
  std::vector<QpExchangeInfo> qpAll(static_cast<size_t>(s.worldSize) * rowLen);
  std::memcpy(&qpAll[static_cast<size_t>(s.rank) * rowLen], qpInfo.data(), qpInfo.size() * sizeof(QpExchangeInfo));
  s.bootstrap->allGather(qpAll.data(), static_cast<int>(rowLen * sizeof(QpExchangeInfo)));

  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int q = 0; q < nQp; ++q) {
      // Remote peer r's matching logical QP that targets this rank.
      const QpExchangeInfo& remote = qpAll[static_cast<size_t>(r) * rowLen + s.qpIndex(s.rank, q)];
      s.connectQp(s.hcaIndex(q), s.qpHl[s.qpIndex(r, q)], remote);
    }
  }

  bool needsCpuProxy = false;
  for (auto* q : s.qpHl) {
    if (q && q->qp_gverbs && q->qp_gverbs->cpu_proxy) {
      needsCpuProxy = true;
      break;
    }
  }
  if (needsCpuProxy) {
    MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_service(&s.cpuProxyService));
    for (auto* q : s.qpHl) {
      if (q && q->qp_gverbs && q->qp_gverbs->cpu_proxy) {
        MSCCLPP_DOCA_THROW(doca_gpu_verbs_service_monitor_qp(s.cpuProxyService, q->qp_gverbs));
      }
    }
  }

  // 6. Flatten the per-peer device QPs into a GPU array (peer-major).
  MSCCLPP_DOCA_THROW(
      doca_gpu_verbs_qp_flat_list_create_hl(s.qpHl.data(), static_cast<uint32_t>(s.qpHl.size()), &s.qpFlatGpu));

  // 7. Exchange one rkey per HCA plus the symmetric base address.
  std::vector<MemExchangeInfo> memAll(static_cast<size_t>(s.worldSize) * nHcas);
  std::memset(memAll.data(), 0, memAll.size() * sizeof(MemExchangeInfo));
  for (int hca = 0; hca < nHcas; ++hca) {
    auto& info = memAll[static_cast<size_t>(s.rank) * nHcas + hca];
    info.base = reinterpret_cast<uint64_t>(symmetricBuffer);
    info.rkey = s.hcas[hca].mr->getInfo().rkey;
  }
  s.bootstrap->allGather(memAll.data(), static_cast<int>(nHcas * sizeof(MemExchangeInfo)));

  std::vector<uint32_t> rkeysHost(static_cast<size_t>(nHcas) * s.worldSize);
  std::vector<uint32_t> lkeysHost(nHcas);
  std::vector<uintptr_t> baseHost(s.worldSize);
  for (int hca = 0; hca < nHcas; ++hca) {
    lkeysHost[hca] = s.hcas[hca].mr->getLkey();
  }
  for (int r = 0; r < s.worldSize; ++r) {
    for (int hca = 0; hca < nHcas; ++hca) {
      const auto& info = memAll[static_cast<size_t>(r) * nHcas + hca];
      rkeysHost[static_cast<size_t>(hca) * s.worldSize + r] = htobe32(info.rkey);
    }
    baseHost[r] = static_cast<uintptr_t>(memAll[static_cast<size_t>(r) * nHcas].base);
  }

  // 8. Publish device-side arrays + context.
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.rkeysGpu, sizeof(uint32_t) * rkeysHost.size()));
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.lkeysGpu, sizeof(uint32_t) * lkeysHost.size()));
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.peerBaseGpu, sizeof(uintptr_t) * s.worldSize));
  MSCCLPP_CUDA_THROW(
      cudaMemcpy(s.rkeysGpu, rkeysHost.data(), sizeof(uint32_t) * rkeysHost.size(), cudaMemcpyHostToDevice));
  MSCCLPP_CUDA_THROW(
      cudaMemcpy(s.lkeysGpu, lkeysHost.data(), sizeof(uint32_t) * lkeysHost.size(), cudaMemcpyHostToDevice));
  MSCCLPP_CUDA_THROW(
      cudaMemcpy(s.peerBaseGpu, baseHost.data(), sizeof(uintptr_t) * s.worldSize, cudaMemcpyHostToDevice));

  GpuNetIoDeviceContext ctxHost{};
  ctxHost.qps = s.qpFlatGpu;
  ctxHost.rkeys = s.rkeysGpu;
  ctxHost.peerBase = s.peerBaseGpu;
  ctxHost.lkey = lkeysHost[0];
  ctxHost.localBase = reinterpret_cast<uintptr_t>(symmetricBuffer);
  ctxHost.numPeers = s.worldSize;
  ctxHost.numQpsPerPeer = s.numQpsPerPeer;
  ctxHost.numHcas = nHcas;
  ctxHost.lkeys = s.lkeysGpu;
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.ctxGpu, sizeof(GpuNetIoDeviceContext)));
  MSCCLPP_CUDA_THROW(cudaMemcpy(s.ctxGpu, &ctxHost, sizeof(GpuNetIoDeviceContext), cudaMemcpyHostToDevice));
}

GpuNetIoDeviceContext* GpuNetIoService::deviceContext() const { return pimpl_->ctxGpu; }

}  // namespace mscclpp
