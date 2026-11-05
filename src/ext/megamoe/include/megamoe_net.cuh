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

// Device side of the inter-rack exchange described in megamoe_net.hpp. Nothing here depends on the routed GEMM
// specialization, so every helper takes explicit dimensions and buffers.
namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int NetSendThreads = 512;
// Source-side combineIn row marker of a route whose owner returns its token's reduced row through another slot.
constexpr int NetZeroRow = -2;
constexpr int64_t NetSpinLimit = 1000000000;

/// The signals a peer sends per launch. Routes travel on queue pair 0 only; rows and results are striped over every
/// queue pair of the pair, each signaling its share. Each queue pair's signals follow this order.
enum class NetPhase : int { Routes = 0, Rows = 1, Results = 2 };

/// Signal count of queue pair `qp` that completes `phase` of launch `epoch`: queue pair 0 carries 3 signals per launch,
/// the others 2.
MSCCLPP_DEVICE_INLINE uint64_t netExpected(int qp, NetPhase phase, uint64_t epoch) {
  const uint64_t perLaunch = qp == 0 ? 3 : 2;
  const uint64_t index = qp == 0 ? uint64_t(phase) : uint64_t(phase) - 1;
  return perLaunch * (epoch - 1) + index + 1;
}

MSCCLPP_DEVICE_INLINE bool netArrived(const NetDevice& net, int peer, int qp, NetPhase phase, uint64_t epoch) {
  return atomicLoad<uint64_t, scopeSystem>(net.channel(qp, peer).semaphore_.inboundToken, memoryOrderAcquire) >=
         netExpected(qp, phase, epoch);
}

/// Wait until `peer` has sent `phase` of launch `epoch` on every queue pair that carries it. Each queue pair's
/// semaphore counts its phases in strict order, so waiting does not consume a signal and any number of threads may
/// wait.
MSCCLPP_DEVICE_INLINE void netWait(const NetDevice& net, int peer, NetPhase phase, uint64_t epoch) {
  const int qps = phase == NetPhase::Routes ? 1 : net.qps;
  for (int qp = 0; qp < qps; ++qp) POLL_MAYBE_JAILBREAK(!netArrived(net, peer, qp, phase, epoch), NetSpinLimit);
}

MSCCLPP_DEVICE_INLINE const NetHeader* netRecvHeader(const NetDevice& net, int index) {
  return reinterpret_cast<const NetHeader*>(net.base + net.layout.recvMeta + size_t(index) * net.layout.metaBytes);
}

MSCCLPP_DEVICE_INLINE const NetEntry* netRecvEntries(const NetDevice& net, int index) {
  return reinterpret_cast<const NetEntry*>(netRecvHeader(net, index) + 1);
}

/// First entry of each received token row of network source `index`, then the entry count: [maxTokens + 1].
MSCCLPP_DEVICE_INLINE int32_t* netRowStart(const NetDevice& net, int index, int maxTokens) {
  return reinterpret_cast<int32_t*>(net.base + net.layout.rowStart) + size_t(index) * (maxTokens + 1);
}

MSCCLPP_DEVICE_INLINE int32_t* netRouteMap(const NetDevice& net, int index, size_t routesPerRank) {
  return reinterpret_cast<int32_t*>(net.base + net.layout.routeMap) + size_t(index) * routesPerRank;
}

/// The received entry of a route from network source `rank`. Valid after routing published the route.
MSCCLPP_DEVICE_INLINE NetEntry netRouteEntry(const NetDevice& net, int rank, int token, int slot, int topK,
                                             size_t routesPerRank) {
  int index = net.indexOf(rank);
  int entry = __ldcg(netRouteMap(net, index, routesPerRank) + size_t(token) * topK + slot);
  int4 raw = __ldcg(reinterpret_cast<const int4*>(netRecvEntries(net, index) + entry));
  return NetEntry{raw.x, raw.y, __int_as_float(raw.z), raw.w};
}

MSCCLPP_DEVICE_INLINE const uint8_t* netRecvRow(const NetDevice& net, int rank, int row, int maxTokens) {
  return net.base + net.layout.recvRows + (size_t(net.indexOf(rank)) * maxTokens + row) * net.layout.rowBytes;
}

/// The owner-side staging row of a route from network source `rank`, returned to the source after FC2.
MSCCLPP_DEVICE_INLINE uint8_t* netCombineRow(const NetDevice& net, int rank, int token, int slot, int topK,
                                             size_t routesPerRank) {
  int index = net.indexOf(rank);
  int entry = __ldcg(netRouteMap(net, index, routesPerRank) + size_t(token) * topK + slot);
  return net.base + net.layout.combineOut + size_t(index) * net.layout.slotBytes + size_t(entry) * net.layout.outBytes;
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

/// Put `rows` rows of `rowBytes` to `peer`, split into near-equal row ranges over its queue pairs, and signal each
/// queue pair (also when its range is empty). `wqeSlot` is the caller's shared-memory BlueFlame slot, or null.
MSCCLPP_DEVICE_INLINE void netPutRowsWithSignal(const NetDevice& net, int peer, uint64_t dst, uint64_t src,
                                                uint64_t rows, uint64_t rowBytes, GpuNetIoWqeSlot* wqeSlot) {
  for (int qp = 0; qp < net.qps; ++qp) {
    const uint64_t begin = rows * qp / net.qps;
    const uint64_t end = rows * (qp + 1) / net.qps;
    NetChannel channel = net.channel(qp, peer);
    if (wqeSlot) channel.setWqeSlot(wqeSlot);
    netPutWithSignal(channel, dst + begin * rowBytes, src + begin * rowBytes, (end - begin) * rowBytes);
  }
}

struct NetSendArguments {
  int rank;
  int maxTokens;
  int topK;
  int localExperts;
  int hidden;
  int scaleStride;  // 0 skips scales (static FP8)
  int nTokens;
  const int32_t* ids;
  const float* weights;
  const uint8_t* input;       // FP8 [nTokens, hidden]
  const uint8_t* inputScale;  // E8M0 [nTokens, scaleStride]
};

// netSendParts(nNet) CTAs per network owner pack and send this rank's message to that owner. Launched on the forward
// stream after input quantization and before the routed kernel, every launch, including empty ones. Part 0 builds the
// header and entries and sends them first, so the owner can route while the token rows are still in flight; every part
// then packs its share of the rows, and part 0 sends them once all parts are done. Only part 0 posts, so each queue
// pair has a single posting thread.
template <int Threads = NetSendThreads>
__global__ void __launch_bounds__(Threads) netSendKernel(NetDevice net, NetSendArguments a) {
  __shared__ GpuNetIoWqeSlot wqeSlot;
  __shared__ int sums[Threads / 32];
  __shared__ int tileTokens[Threads];
  const int parts = gridDim.x / net.nNet;
  const int index = blockIdx.x / parts;
  const int part = blockIdx.x % parts;
  const int owner = net.rankOf(index);
  const int nRoutes = a.nTokens * a.topK;
  auto ownerOf = [&](int id) {
    int rank = id >= 0 ? id / a.localExperts : -1;
    return rank < net.worldSize ? rank : -1;
  };
  uint8_t* rows = net.base + net.layout.sendRows + size_t(index) * a.maxTokens * net.layout.rowBytes;
  const int sourceIndex = net.indexFrom(owner, a.rank);
  auto send = [&](uint64_t dst, uint64_t src, uint64_t bytes) {
    NetChannel channel = net.channel(0, owner);
    channel.setWqeSlot(&wqeSlot);
    netPutWithSignal(channel, dst, src, bytes);
  };
  // Number of token rows this rank sends `owner`, the routes of each token to it, and the token's first such slot.
  auto ownedBy = [&](int token, int rank, int& first) {
    int owned = 0;
    first = -1;
    if (token < a.nTokens) {
      for (int slot = 0; slot < a.topK; ++slot) {
        if (ownerOf(a.ids[token * a.topK + slot]) != rank) continue;
        if (!owned) first = slot;
        ++owned;
      }
    }
    return owned;
  };

  if (part == 0) {
    uint8_t* message = net.base + net.layout.sendMeta + size_t(index) * net.layout.metaBytes;
    auto* header = reinterpret_cast<NetHeader*>(message);
    auto* entries = reinterpret_cast<NetEntry*>(header + 1);
    auto* routeRows = reinterpret_cast<int32_t*>(net.base + net.layout.routeRows);
    // Owners' results are packed into combineIn in network-index order: one row per route, or with owner-side
    // reduction one row per token row.
    int below = 0;
    for (int route = threadIdx.x; route < nRoutes; route += Threads) {
      int rank = ownerOf(a.ids[route]);
      bool netOwned = rank >= 0 && net.isNet(rank);
      if (netOwned && net.indexOf(rank) < index) {
        int first = route % a.topK;
        if (net.reduce) ownedBy(route / a.topK, rank, first);
        below += first == route % a.topK;
      }
      if (index == 0 && !netOwned) routeRows[route] = -1;
    }
    int outBase;
    (void)netBlockExclusiveScan<Threads>(below, outBase, sums);
    int nEntries = 0;
    int nRows = 0;
    for (int firstToken = 0; firstToken < a.nTokens; firstToken += Threads) {
      const int token = firstToken + threadIdx.x;
      int firstSlot;
      const int owned = ownedBy(token, owner, firstSlot);
      int tileEntries;
      int tileRows;
      int entry = nEntries + netBlockExclusiveScan<Threads>(owned, tileEntries, sums);
      const int row = nRows + netBlockExclusiveScan<Threads>(owned ? 1 : 0, tileRows, sums);
      for (int slot = firstSlot; owned && slot < a.topK; ++slot) {
        int route = token * a.topK + slot;
        int id = a.ids[route];
        if (ownerOf(id) != owner) continue;
        entries[entry] = NetEntry{route, id - owner * a.localExperts, a.weights[route], row};
        routeRows[route] = net.reduce ? (slot == firstSlot ? outBase + row : NetZeroRow) : outBase + entry;
        ++entry;
      }
      nEntries += tileEntries;
      nRows += tileRows;
    }
    if (threadIdx.x == 0) *header = NetHeader{a.nTokens, nEntries, outBase, nRows};
    __syncthreads();
    if (threadIdx.x == 0) {
      // The NIC reads the message written by every thread of this CTA.
      __threadfence_system();
      send(net.layout.recvMeta + size_t(sourceIndex) * net.layout.metaBytes,
           net.layout.sendMeta + size_t(index) * net.layout.metaBytes,
           sizeof(NetHeader) + size_t(nEntries) * sizeof(NetEntry));
    }
  }

  // Every part packs its share of each tile's token rows.
  constexpr int VectorBytes = 16;
  const int inputVectors = a.hidden / VectorBytes;
  const int rowVectors = inputVectors + a.scaleStride / VectorBytes;
  int nRows = 0;
  for (int firstToken = 0; firstToken < a.nTokens; firstToken += Threads) {
    const int token = firstToken + threadIdx.x;
    int firstSlot;
    const bool owned = ownedBy(token, owner, firstSlot) > 0;
    int tileRows;
    const int row = netBlockExclusiveScan<Threads>(owned ? 1 : 0, tileRows, sums);
    if (owned) tileTokens[row] = token;
    __syncthreads();
    for (int item = part * Threads + threadIdx.x; item < tileRows * rowVectors; item += parts * Threads) {
      int tileRow = item / rowVectors;
      int vector = item % rowVectors;
      int sourceToken = tileTokens[tileRow];
      const uint8_t* source =
          vector < inputVectors
              ? a.input + size_t(sourceToken) * a.hidden + size_t(vector) * VectorBytes
              : a.inputScale + size_t(sourceToken) * a.scaleStride + size_t(vector - inputVectors) * VectorBytes;
      *reinterpret_cast<int4*>(rows + size_t(nRows + tileRow) * net.layout.rowBytes + size_t(vector) * VectorBytes) =
          *reinterpret_cast<const int4*>(source);
    }
    nRows += tileRows;
    // The next tile reuses tileTokens.
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int* done = reinterpret_cast<int*>(net.base + net.layout.sendDone) + index;
    if (part) {
      __threadfence();
      atomicAdd(done, 1);
    } else {
      POLL_MAYBE_JAILBREAK((atomicLoad<int, scopeDevice>(done, memoryOrderAcquire) < parts - 1), NetSpinLimit);
      *done = 0;
      __threadfence_system();
      netPutRowsWithSignal(net, owner, net.layout.recvRows + size_t(sourceIndex) * a.maxTokens * net.layout.rowBytes,
                           net.layout.sendRows + size_t(index) * a.maxTokens * net.layout.rowBytes, nRows,
                           net.layout.rowBytes, &wqeSlot);
    }
  }
}

// Index of the first entry whose route is at least `route`; entries are in route order.
MSCCLPP_DEVICE_INLINE int netLowerBound(const NetEntry* entries, int nEntries, int route) {
  int low = 0;
  int high = nEntries;
  while (low < high) {
    int middle = (low + high) / 2;
    if (__ldcg(&entries[middle].route) < route)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
}

struct NetRouting {
  mscclpp::LL8Packet* tokenCount;  // [worldSize], or null with fixed token counts
  mscclpp::LL8Packet* headers;     // [worldSize][chunksPerRank]
  mscclpp::LL8Packet* packets;     // [worldSize][routesPerRank]
  int chunksPerRank;
  int maxTokens;
  int topK;
};

/// Called by every thread of a CTA. Waits for the routes of network source `index` and republishes them as the
/// current-epoch LL8 token count, chunk headers, and route packets (local expert << 16 | offset in its chunk of
/// RouteChunk routes) that an NVLink source writes remotely. Also records each route's entry for dispatch and FC2 and
/// each token row's first entry for the return, then waits for the token rows, which travel behind the routes.
template <int RouteChunk>
MSCCLPP_DEVICE_INLINE void netRelaySource(const NetDevice& net, const NetRouting& r, int index, uint64_t epoch) {
  const int source = net.rankOf(index);
  const size_t routesPerRank = size_t(r.maxTokens) * r.topK;
  if (threadIdx.x == 0) {
    netWait(net, source, NetPhase::Routes, epoch);
  }
  __syncthreads();
  const NetHeader* header = netRecvHeader(net, index);
  const NetEntry* entries = netRecvEntries(net, index);
  const int nTokens = __ldcg(&header->nTokens);
  const int nEntries = __ldcg(&header->nEntries);
  if (r.tokenCount && threadIdx.x == 0) r.tokenCount[source].write(uint32_t(nTokens), uint32_t(epoch));
  int32_t* routeMap = netRouteMap(net, index, routesPerRank);
  int32_t* rowStart = netRowStart(net, index, r.maxTokens);
  if (threadIdx.x == 0) rowStart[__ldcg(&header->nRows)] = nEntries;
  mscclpp::LL8Packet* packets = r.packets + size_t(source) * routesPerRank;
  for (int entry = threadIdx.x; entry < nEntries; entry += blockDim.x) {
    int route = __ldcg(&entries[entry].route);
    int expert = __ldcg(&entries[entry].expert);
    int chunk = route / RouteChunk;
    int position = entry - netLowerBound(entries, nEntries, chunk * RouteChunk);
    routeMap[route] = entry;
    const int row = __ldcg(&entries[entry].row);
    if (entry == 0 || __ldcg(&entries[entry - 1].row) != row) rowStart[row] = entry;
    packets[size_t(chunk) * RouteChunk + position].write(uint32_t(expert) << 16 | uint32_t(route % RouteChunk),
                                                         uint32_t(epoch));
  }
  const int nRoutes = nTokens * r.topK;
  const int nChunks = (nRoutes + RouteChunk - 1) / RouteChunk;
  for (int chunk = threadIdx.x; chunk < nChunks; chunk += blockDim.x) {
    int count = netLowerBound(entries, nEntries, (chunk + 1) * RouteChunk) -
                netLowerBound(entries, nEntries, chunk * RouteChunk);
    r.headers[size_t(source) * r.chunksPerRank + chunk].write(uint32_t(count), uint32_t(epoch));
  }
  // Dispatch reads the rows; waiting here lets the relay overlap their transfer without stalling the GEMM pipeline.
  if (threadIdx.x == 0) netWait(net, source, NetPhase::Rows, epoch);
}

/// Called by block 0 before the grid barrier that precedes netReturnResults: copy each network source's header to
/// local memory. Once a source receives its results it may send its next message, which overwrites the received
/// header, so the return reads only this copy.
MSCCLPP_DEVICE_INLINE void netSnapshotHeaders(const NetDevice& net, uint64_t epoch) {
  auto* info = reinterpret_cast<NetHeader*>(net.base + net.layout.recvInfo);
  for (int index = threadIdx.x; index < net.nNet; index += blockDim.x) {
    // Routing skips the relay when every rank is empty, so the message may not have been observed yet.
    netWait(net, net.rankOf(index), NetPhase::Routes, epoch);
    const NetHeader* header = netRecvHeader(net, index);
    info[index] = NetHeader{__ldcg(&header->nTokens), __ldcg(&header->nEntries), __ldcg(&header->outBase),
                            __ldcg(&header->nRows)};
  }
}

/// Called by every thread of every CTA after netSnapshotHeaders and after every FC2 store of this launch has completed
/// grid-wide: return each network source's results and signal it. With owner-side reduction, warps sum the weighted
/// rows of each received token row into reduceOut, source by source, and the warp completing a source's last row sends
/// that source's rows; otherwise block 0 sends the per-route rows staged in combineOut. Exactly one thread posts per
/// source. Row boundaries come from the relay's local rowStart, never from the received message.
MSCCLPP_DEVICE_INLINE void netReturnResults(const NetDevice& net) {
  const auto* info = reinterpret_cast<const NetHeader*>(net.base + net.layout.recvInfo);
  auto post = [&](int index, uint64_t src, uint64_t rows) {
    netPutRowsWithSignal(net, net.rankOf(index),
                         net.layout.combineIn + size_t(info[index].outBase) * net.layout.outBytes, src, rows,
                         net.layout.outBytes, nullptr);
  };
  if (!net.reduce) {
    for (int index = threadIdx.x; blockIdx.x == 0 && index < net.nNet; index += blockDim.x) {
      __threadfence_system();
      post(index, net.layout.combineOut + size_t(index) * net.layout.slotBytes, info[index].nEntries);
    }
    return;
  }
  // Work items are 128-vector segments of reduced rows, spread over every warp of the grid so that the latency-bound
  // loads of small launches run in parallel. A source's results go out once all of its segments are written.
  constexpr int LaneVectors = 4;
  constexpr int SegmentVectors = 32 * LaneVectors;
  const int lane = threadIdx.x % 32;
  const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int vectors = int(net.layout.outBytes / sizeof(int4));
  const int segments = (vectors + SegmentVectors - 1) / SegmentVectors;
  auto* done = reinterpret_cast<int*>(net.base + net.layout.returnDone);
  int items = 0;
  for (int index = 0; index < net.nNet; ++index) {
    items += info[index].nRows * segments;
    if (!info[index].nRows && warp == 0 && lane == 0)
      post(index, net.layout.reduceOut + size_t(index) * net.layout.reduceSlotBytes, 0);
  }
  int index = 0;
  int base = 0;
  for (int item = warp; item < items; item += warps) {
    while (item >= base + info[index].nRows * segments) base += info[index++].nRows * segments;
    const int nRows = info[index].nRows;
    const int row = (item - base) / segments;
    const int first = (item - base) % segments * SegmentVectors + lane;
    const int32_t* rowStart = netRowStart(net, index, net.maxTokens);
    const int begin = rowStart[row];
    const int end = rowStart[row + 1];
    const uint64_t reduced = net.layout.reduceOut + size_t(index) * net.layout.reduceSlotBytes;
    const auto* slot =
        reinterpret_cast<const int4*>(net.base + net.layout.combineOut + size_t(index) * net.layout.slotBytes);
    auto* out = reinterpret_cast<int4*>(net.base + reduced + size_t(row) * net.layout.outBytes);
    float2 sum[LaneVectors][4] = {};
    for (int entry = begin; entry < end; ++entry) {
      int4 raw[LaneVectors];
#pragma unroll
      for (int v = 0; v < LaneVectors; ++v) {
        const int vector = first + v * 32;
        raw[v] = vector < vectors ? __ldcg(slot + size_t(entry) * vectors + vector) : make_int4(0, 0, 0, 0);
      }
#pragma unroll
      for (int v = 0; v < LaneVectors; ++v) {
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw[v]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 value = __bfloat1622float2(pairs[i]);
          sum[v][i].x += value.x;
          sum[v][i].y += value.y;
        }
      }
    }
#pragma unroll
    for (int v = 0; v < LaneVectors; ++v) {
      const int vector = first + v * 32;
      if (vector >= vectors) continue;
      int4 packed;
      auto* outPairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) outPairs[i] = __float22bfloat162_rn(sum[v][i]);
      out[vector] = packed;
    }
    __threadfence();
    __syncwarp();
    if (lane == 0 && atomicAdd(done + index, 1) == nRows * segments - 1) {
      done[index] = 0;
      __threadfence_system();
      post(index, reduced, nRows);
    }
  }
}

/// Called by every thread of every CTA: wait for every network owner's results, then copy this CTA's share of them
/// into the partial output rows [route][outBytes]. A route whose token row was reduced into another slot of the same
/// owner gets a zero row. The caller synchronizes the grid before reading the partials.
MSCCLPP_DEVICE_INLINE void netCollectResults(const NetDevice& net, uint8_t* partialOutput, int nRoutes,
                                             uint64_t epoch) {
  for (int index = threadIdx.x; index < net.nNet; index += blockDim.x) {
    netWait(net, net.rankOf(index), NetPhase::Results, epoch);
  }
  __syncthreads();
  const auto* routeRows = reinterpret_cast<const int32_t*>(net.base + net.layout.routeRows);
  const uint8_t* combineIn = net.base + net.layout.combineIn;
  const int lane = threadIdx.x % 32;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int vectors = int(net.layout.outBytes / sizeof(int4));
  for (int route = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; route < nRoutes; route += warps) {
    int row = routeRows[route];
    if (row == -1) continue;
    auto* destination = reinterpret_cast<int4*>(partialOutput + size_t(route) * net.layout.outBytes);
    if (row == NetZeroRow) {
      for (int vector = lane; vector < vectors; vector += 32) destination[vector] = make_int4(0, 0, 0, 0);
      continue;
    }
    auto* source = reinterpret_cast<const int4*>(combineIn + size_t(row) * net.layout.outBytes);
    for (int vector = lane; vector < vectors; vector += 32) destination[vector] = __ldcg(source + vector);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_NET_CUH_
