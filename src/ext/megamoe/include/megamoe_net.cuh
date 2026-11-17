// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_NET_CUH_
#define MSCCLPP_EXT_MEGAMOE_NET_CUH_

#include <cuda_bf16.h>

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/poll_device.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "megamoe_net.hpp"

// Device side of the rail-aggregated inter-rack exchange described in megamoe_net.hpp. Nothing here depends on the
// routed GEMM specialization, so every helper takes explicit dimensions and buffers.
namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int64_t NetSpinLimit = 1000000000;

/// The signals a rail peer sends per launch. Routes travel on queue pair 0; rows on the other queue pairs (or on queue
/// pair 0 if it is the only one), so they leave while the routes are built; results on every queue pair. Each queue
/// pair signals its share of a phase it carries, also when empty, in this order.
enum class NetPhase : int { Routes = 0, Rows = 1, Results = 2 };

/// First queue pair carrying rows.
MSCCLPP_HOST_DEVICE_INLINE int netFirstRowQp(int qps) { return qps > 1 ? 1 : 0; }

MSCCLPP_DEVICE_INLINE bool netCarries(int qps, int qp, NetPhase phase) {
  if (phase == NetPhase::Routes) return qp == 0;
  if (phase == NetPhase::Rows) return qp >= netFirstRowQp(qps);
  return true;
}

/// Signal count of queue pair `qp` that completes `phase` of launch `epoch`, counting the phases the queue pair
/// carries.
MSCCLPP_DEVICE_INLINE uint64_t netExpected(int qps, int qp, NetPhase phase, uint64_t epoch) {
  uint64_t perLaunch = 0;
  uint64_t index = 0;
  for (int other = 0; other < 3; ++other) {
    if (!netCarries(qps, qp, NetPhase(other))) continue;
    if (other <= int(phase)) index = perLaunch + 1;
    ++perLaunch;
  }
  return perLaunch * (epoch - 1) + index;
}

MSCCLPP_DEVICE_INLINE bool netArrived(const NetDevice& net, int index, int qp, NetPhase phase, uint64_t epoch) {
  return atomicLoad<uint64_t, scopeSystem>(net.channel(qp, index).semaphore_.inboundToken, memoryOrderAcquire) >=
         netExpected(net.qps, qp, phase, epoch);
}

/// Wait until the rail peer of remote rack `index` has sent `phase` of launch `epoch` on every queue pair that carries
/// it. Waiting does not consume a signal, so any number of threads may wait.
MSCCLPP_DEVICE_INLINE void netWait(const NetDevice& net, int index, NetPhase phase, uint64_t epoch) {
  for (int qp = 0; qp < net.qps; ++qp)
    if (netCarries(net.qps, qp, phase)) POLL_MAYBE_JAILBREAK(!netArrived(net, index, qp, phase, epoch), NetSpinLimit);
}

MSCCLPP_DEVICE_INLINE size_t netEntriesOffset(int nChunks, int rackSize) {
  return sizeof(NetHeader) + (size_t(nChunks) * rackSize * sizeof(int32_t) + 15) / 16 * 16;
}

// Exclusive prefix over a block of Threads threads; `sums` is shared scratch of Threads / 32 ints.
template <int Threads>
MSCCLPP_DEVICE_INLINE int netBlockExclusiveScan(int value, int& total, int* sums) {
  static_assert(Threads % 32 == 0 && Threads <= 1024);
  constexpr int Warps = Threads / 32;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  int inclusive = value;
#pragma unroll
  for (int distance = 1; distance < 32; distance *= 2) {
    int other = __shfl_up_sync(0xffffffff, inclusive, distance);
    if (lane >= distance) inclusive += other;
  }
  if (lane == 31) sums[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int sum = lane < Warps ? sums[lane] : 0;
#pragma unroll
    for (int distance = 1; distance < 32; distance *= 2) {
      int other = __shfl_up_sync(0xffffffff, sum, distance);
      if (lane >= distance) sum += other;
    }
    if (lane < Warps) sums[lane] = sum;
  }
  __syncthreads();
  int prefix = warp ? sums[warp - 1] : 0;
  total = sums[Warps - 1];
  // The caller may reuse the scratch immediately.
  __syncthreads();
  return prefix + inclusive - value;
}

// Put `bytes` and signal the peer; an empty transfer only signals.
MSCCLPP_DEVICE_INLINE void netPutWithSignal(NetChannel& channel, uint64_t dst, uint64_t src, uint64_t bytes) {
  if (!bytes) {
    channel.signal<PortChannelSharingMode::Exclusive>();
    return;
  }
  channel.putWithSignal<PortChannelSharingMode::Exclusive>(dst, src, bytes);
}

/// Queue pair `qp`'s share of `rows` rows of `rowBytes` to the rail peer of remote rack `index`: the rows are split
/// into near-equal ranges over as many queue pairs from `firstQp` on as carry at least NetMinQpBytes each, and every
/// such queue pair signals, also when its range is empty. Callers post the queue pairs from separate warps, so their
/// doorbells overlap. `wqeSlot` is the caller's shared-memory BlueFlame slot, or null.
MSCCLPP_DEVICE_INLINE void netPutRowsWithSignal(const NetDevice& net, int index, int qp, uint64_t dst, uint64_t src,
                                                uint64_t rows, uint64_t rowBytes, GpuNetIoWqeSlot* wqeSlot,
                                                int firstQp = 0) {
  const uint64_t used = min(uint64_t(net.qps - firstQp), max(uint64_t(1), rows * rowBytes / NetMinQpBytes));
  const uint64_t share = qp - firstQp;
  const uint64_t begin = share < used ? rows * share / used : rows;
  const uint64_t end = share < used ? rows * (share + 1) / used : rows;
  NetChannel channel = net.channel(qp, index);
  if (wqeSlot) channel.setWqeSlot(wqeSlot);
  netPutWithSignal(channel, dst + begin * rowBytes, src + begin * rowBytes, (end - begin) * rowBytes);
}

/// Reserve consecutive positions from a shared counter per key, one atomic per group of lanes sharing a key. All lanes
/// of the warp must call this; lanes with a negative key receive -1.
MSCCLPP_DEVICE_INLINE int netReserveGroup(int* counters, int key, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, key);
  if (key < 0) return -1;
  int leader = __ffs(group) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(counters + key, __popc(group));
  base = __shfl_sync(group, base, leader);
  return base + __popc(group & ((1u << lane) - 1));
}

struct NetSource {
  int nTokens;
  int topK;
  int localExperts;
  const int32_t* ids;
  const float* weights;
};

/// Called by every thread of a CTA of Threads == RouteChunk threads at the start of routing: build this rank's message
/// for the remote rack of network index `index`, put it to the rail peer there and signal (Routes); with a single queue
/// pair, then put every token row and signal (Rows). `scratch` is shared memory of at least rackSize + Threads / 32
/// ints; `wqeSlots` are MaxNetQps shared BlueFlame slots. Messages are sent every launch, including empty ones, since
/// peers count signals per launch.
template <int Threads>
MSCCLPP_DEVICE_INLINE void netSendRoutes(const NetDevice& net, int index, const NetSource& a, int* scratch,
                                         GpuNetIoWqeSlot* wqeSlots) {
  const int rackSize = net.rackSize;
  const int remoteRack = net.rackOfIndex(index);
  const int lane = threadIdx.x % 32;
  int* ownerCounts = scratch;
  int* sums = scratch + rackSize;
  uint8_t* message = net.base + net.layout.sendMeta + size_t(index) * net.layout.metaBytes;
  auto* header = reinterpret_cast<NetHeader*>(message);
  auto* counts = reinterpret_cast<int32_t*>(header + 1);
  const int nRoutes = a.nTokens * a.topK;
  const int nChunks = nRoutes ? (nRoutes + Threads - 1) / Threads : 0;
  auto* entries = reinterpret_cast<NetEntry*>(message + netEntriesOffset(nChunks, rackSize));
  const int at = net.indexAt(index);
  const int firstRowQp = netFirstRowQp(net.qps);
  for (int i = threadIdx.x; i < rackSize; i += Threads) ownerCounts[i] = 0;
  __syncthreads();
  int nEntries = 0;
  for (int chunk = 0; chunk < nChunks; ++chunk) {
    const int route = chunk * Threads + threadIdx.x;
    const int id = route < nRoutes ? a.ids[route] : -1;
    const int owner = id >= 0 ? id / a.localExperts : -1;
    const int ownerLane = owner >= 0 && owner / rackSize == remoteRack ? owner % rackSize : -1;
    const int position = netReserveGroup(ownerCounts, ownerLane, lane);
    int total;
    const int entry = nEntries + netBlockExclusiveScan<Threads>(ownerLane >= 0, total, sums);
    if (ownerLane >= 0)
      entries[entry] =
          NetEntry{route, uint16_t(id - owner * a.localExperts), uint16_t(position), a.weights[route], ownerLane};
    for (int i = threadIdx.x; i < rackSize; i += Threads) {
      counts[chunk * rackSize + i] = ownerCounts[i];
      ownerCounts[i] = 0;
    }
    nEntries += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) *header = NetHeader{a.nTokens, nEntries, nChunks, 0};
  __syncthreads();
  if (threadIdx.x == 0) {
    // The NIC reads the message written by every thread of this CTA.
    __threadfence_system();
    NetChannel channel = net.channel(0, index);
    channel.setWqeSlot(wqeSlots);
    netPutWithSignal(channel, net.layout.recvMeta + size_t(at) * net.layout.metaBytes,
                     net.layout.sendMeta + size_t(index) * net.layout.metaBytes,
                     netEntriesOffset(nChunks, rackSize) + size_t(nEntries) * sizeof(NetEntry));
    // With a single queue pair, the rows follow the routes on it.
    if (!firstRowQp)
      netPutRowsWithSignal(net, index, 0, net.layout.recvRows + size_t(at) * net.maxTokens * net.layout.rowBytes,
                           net.layout.sendRows, a.nTokens, net.layout.rowBytes, wqeSlots);
  }
  __syncthreads();
}

/// Called by every thread of a CTA at the start of routing when rows have their own queue pairs: warp q posts queue
/// pair q's share of every token row to the rail peer of remote rack `index` and signals (Rows). The input quantization
/// wrote the rows before this kernel, so they leave while another CTA builds the routes. Separate warps post the queue
/// pairs so their doorbells overlap. `wqeSlots` are MaxNetQps shared BlueFlame slots.
MSCCLPP_DEVICE_INLINE void netSendRows(const NetDevice& net, int index, int nTokens, GpuNetIoWqeSlot* wqeSlots) {
  const int qp = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0 && qp >= netFirstRowQp(net.qps) && qp < net.qps)
    netPutRowsWithSignal(net, index, qp,
                         net.layout.recvRows + size_t(net.indexAt(index)) * net.maxTokens * net.layout.rowBytes,
                         net.layout.sendRows, nTokens, net.layout.rowBytes, wqeSlots + qp, netFirstRowQp(net.qps));
  __syncthreads();
}

struct NetRouting {
  const uint64_t* peers;  // symmetric workspace bases by rank
  size_t tokenCount;      // LL8 [worldSize], or 0 with fixed token counts
  size_t headers;         // LL8 [worldSize][chunksPerRank]
  size_t packets;         // LL8 [worldSize][routesPerRank]
  int chunksPerRank;
  int routesPerRank;
};

MSCCLPP_DEVICE_INLINE mscclpp::LL8Packet* netPeerPackets(const NetRouting& r, int rank, size_t offset) {
  return reinterpret_cast<mscclpp::LL8Packet*>(r.peers[rank] + offset);
}

/// Called by every thread of a CTA: wait for the routes of the rail peer in the remote rack of network index `index`
/// and republish them over NVLink to every owner of this rack as the current-epoch LL8 token count, chunk headers, and
/// route packets the source would have written itself, plus the weight of each route. The rows travel behind the
/// routes; dispatch waits for them per source.
template <int RouteChunk>
MSCCLPP_DEVICE_INLINE void netRelaySource(const NetDevice& net, const NetRouting& r, int index, uint64_t epoch) {
  const int source = net.railPeer(index);
  const int rackSize = net.rackSize;
  if (threadIdx.x == 0) netWait(net, index, NetPhase::Routes, epoch);
  __syncthreads();
  const uint8_t* message = net.base + net.layout.recvMeta + size_t(index) * net.layout.metaBytes;
  const auto* header = reinterpret_cast<const NetHeader*>(message);
  const int nTokens = __ldcg(&header->nTokens);
  const int nEntries = __ldcg(&header->nEntries);
  const int nChunks = __ldcg(&header->nChunks);
  const auto* counts = reinterpret_cast<const int32_t*>(header + 1);
  const auto* entries = reinterpret_cast<const int4*>(message + netEntriesOffset(nChunks, rackSize));
  if (r.tokenCount) {
    for (int lane = threadIdx.x; lane < rackSize; lane += blockDim.x)
      netPeerPackets(r, net.rackBegin + lane, r.tokenCount)[source].write(uint32_t(nTokens), uint32_t(epoch));
  }
  const size_t weightBase = size_t(net.sourceIndex(source)) * r.routesPerRank;
  for (int entry = threadIdx.x; entry < nEntries; entry += blockDim.x) {
    const int4 raw = __ldcg(entries + entry);
    const int route = raw.x;
    const uint32_t expert = uint32_t(raw.y) & 0xffffu;
    const int position = int(uint32_t(raw.y) >> 16);
    const int ownerLane = raw.w;
    auto* weights = reinterpret_cast<mscclpp::LL8Packet*>(net.rackBase(ownerLane) + net.layout.weights);
    weights[weightBase + route].write(uint32_t(raw.z), uint32_t(epoch));
    netPeerPackets(r, net.rackBegin + ownerLane,
                   r.packets)[size_t(source) * r.routesPerRank + route / RouteChunk * RouteChunk + position]
        .write(expert << 16 | uint32_t(route % RouteChunk), uint32_t(epoch));
  }
  for (int item = threadIdx.x; item < nChunks * rackSize; item += blockDim.x) {
    const int chunk = item / rackSize;
    const int ownerLane = item % rackSize;
    netPeerPackets(r, net.rackBegin + ownerLane, r.headers)[size_t(source) * r.chunksPerRank + chunk].write(
        uint32_t(__ldcg(counts + item)), uint32_t(epoch));
  }
}

/// Called by every thread of every CTA at the start of the launch: zero the accum rows of this launch's parity that
/// launch epoch - 2 used. Routing's grid barriers order the zeroing before the agents publish the rows.
MSCCLPP_DEVICE_INLINE void netZeroAccum(const NetDevice& net, uint64_t epoch) {
  const int parity = int(epoch & 1);
  const auto* used = reinterpret_cast<const int32_t*>(net.base + net.layout.accumTokens) + parity * net.nNet;
  const size_t vectorsPerRow = net.layout.outBytes / sizeof(int4);
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (int index = 0; index < net.nNet; ++index) {
    auto* rows = reinterpret_cast<int4*>(net.base + net.layout.accum +
                                         (size_t(parity) * net.nNet + index) * net.maxTokens * net.layout.outBytes);
    const size_t vectors = size_t(used[index]) * vectorsPerRow;
    for (size_t vector = size_t(blockIdx.x) * blockDim.x + threadIdx.x; vector < vectors; vector += stride)
      rows[vector] = make_int4(0, 0, 0, 0);
  }
}

/// Dispatch of a network source's route: wait until its forwarder has published the source's rows of this launch.
MSCCLPP_DEVICE_INLINE void netWaitRows(const NetDevice& net, int source, uint64_t epoch) {
  auto* ready = reinterpret_cast<uint64_t*>(net.base + net.layout.rowsReady) + net.sourceIndex(source);
  POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(ready, memoryOrderAcquire) < epoch), NetSpinLimit);
  // Bulk copies read the rows through the async proxy.
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

/// The token row (FP8 activations, then scales) of a network source, in its forwarder's recvRows.
MSCCLPP_DEVICE_INLINE const uint8_t* netRow(const NetDevice& net, int source, int token) {
  return net.forwarderBase(source) + net.layout.recvRows +
         (size_t(net.rackIndex(source / net.rackSize)) * net.maxTokens + token) * net.layout.rowBytes;
}

/// The router weight of a network source's route, relayed by its forwarder.
MSCCLPP_DEVICE_INLINE float netRouteWeight(const NetDevice& net, int source, int route, int routesPerRank,
                                           uint64_t epoch) {
  const auto* weights = reinterpret_cast<const mscclpp::LL8Packet*>(net.base + net.layout.weights);
  return __uint_as_float(
      weights[size_t(net.sourceIndex(source)) * routesPerRank + route].read(uint32_t(epoch), NetSpinLimit));
}

/// The forwarder's accum row of a network source's token, the target of this rank's FC2 bulk reductions.
MSCCLPP_DEVICE_INLINE __nv_bfloat16* netAccumRow(const NetDevice& net, int source, int token, uint64_t epoch) {
  return reinterpret_cast<__nv_bfloat16*>(
      net.forwarderBase(source) + net.layout.accum +
      ((size_t(epoch & 1) * net.nNet + net.rackIndex(source / net.rackSize)) * net.maxTokens + token) *
          net.layout.outBytes);
}

/// Called by one warp per CTA after every FC2 bulk reduction of the CTA has completed: the last CTA of this rank tells
/// every forwarder of the rack, one lane each, that this rank's contributions of the launch are in place.
MSCCLPP_DEVICE_INLINE void netEpilogueDone(const NetDevice& net) {
  const int lane = threadIdx.x % 32;
  bool last = false;
  if (lane == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence_system();
    auto* done = reinterpret_cast<uint32_t*>(net.base + net.layout.epilogueDone);
    last = atomicFetchAdd<uint32_t, scopeDevice>(done, 1u, memoryOrderAcqRel) == gridDim.x - 1;
    if (last) {
      atomicStore<uint32_t, scopeDevice>(done, 0u, memoryOrderRelaxed);
      __threadfence_system();
    }
  }
  if (!__shfl_sync(0xffffffff, last, 0)) return;
  // Lane 0's fence orders every CTA's reductions, which it acquired through the counter, before these increments.
  for (int peerLane = lane; peerLane < net.rackSize; peerLane += 32)
    atomicFetchAdd<uint64_t, scopeSystem>(reinterpret_cast<uint64_t*>(net.rackBase(peerLane) + net.layout.combineDone),
                                          uint64_t(1), memoryOrderRelaxed);
}

/// The network agent of the rail peer in the remote rack of network index `index`, run by one warp that is otherwise
/// idle while the GEMM runs: once the source's rows arrived, it lets the rack's owners dispatch them.
MSCCLPP_DEVICE_INLINE void netAgent(const NetDevice& net, int index, uint64_t epoch) {
  const int lane = threadIdx.x % 32;
  const int source = net.railPeer(index);
  for (int qp = netFirstRowQp(net.qps) + lane; qp < net.qps; qp += 32)
    POLL_MAYBE_JAILBREAK(!netArrived(net, index, qp, NetPhase::Rows, epoch), NetSpinLimit);
  __syncwarp();
  __threadfence_system();
  for (int ownerLane = lane; ownerLane < net.rackSize; ownerLane += 32)
    atomicStore<uint64_t, scopeSystem>(
        reinterpret_cast<uint64_t*>(net.rackBase(ownerLane) + net.layout.rowsReady) + net.sourceIndex(source), epoch,
        memoryOrderRelease);
}

/// Called by every thread of the CTA serving the rail peer of remote rack `index` once its roles have finished (at the
/// entry register count, which GPUNetIO posting needs): once every owner of the rack has drained its FC2 reductions,
/// return the source's rack partials and signal the source. Warp q posts queue pair q, so their doorbells overlap.
/// `wqeSlots` are MaxNetQps shared BlueFlame slots.
MSCCLPP_DEVICE_INLINE void netReturnResults(const NetDevice& net, int index, uint64_t epoch,
                                            GpuNetIoWqeSlot* wqeSlots) {
  const int lane = threadIdx.x % 32;
  const int qp = threadIdx.x / 32;
  if (qp >= net.qps) return;
  const auto* header =
      reinterpret_cast<const NetHeader*>(net.base + net.layout.recvMeta + size_t(index) * net.layout.metaBytes);
  // Read before returning the results: the source overwrites the message once it has them. Routing relayed it.
  const int nTokens = __ldcg(&header->nTokens);
  const int parity = int(epoch & 1);
  if (lane == 0) {
    auto* done = reinterpret_cast<uint64_t*>(net.base + net.layout.combineDone);
    POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(done, memoryOrderAcquire) < epoch * net.rackSize),
                         NetSpinLimit);
    if (qp == 0) reinterpret_cast<int32_t*>(net.base + net.layout.accumTokens)[parity * net.nNet + index] = nTokens;
    netPutRowsWithSignal(net, index, qp,
                         net.layout.combineIn + size_t(net.indexAt(index)) * net.maxTokens * net.layout.outBytes,
                         net.layout.accum + (size_t(parity) * net.nNet + index) * net.maxTokens * net.layout.outBytes,
                         nTokens, net.layout.outBytes, wqeSlots + qp);
  }
  __syncwarp();
}

/// Called by every thread of a CTA before the final combine: wait for every remote rack's partial sums.
MSCCLPP_DEVICE_INLINE void netWaitResults(const NetDevice& net, uint64_t epoch) {
  for (int item = threadIdx.x; item < net.nNet * net.qps; item += blockDim.x)
    POLL_MAYBE_JAILBREAK(!netArrived(net, item / net.qps, item % net.qps, NetPhase::Results, epoch), NetSpinLimit);
}

/// Remote rack `index`'s partial sum of this rank's token `token`.
MSCCLPP_DEVICE_INLINE const uint4* netCombineIn(const NetDevice& net, int index, int token) {
  return reinterpret_cast<const uint4*>(net.base + net.layout.combineIn +
                                        (size_t(index) * net.maxTokens + token) * net.layout.outBytes);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_NET_CUH_
