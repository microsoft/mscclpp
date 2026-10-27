// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_NET_CUH_
#define MSCCLPP_EXT_MEGAMOE_NET_CUH_

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/poll_device.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "megamoe_net.hpp"

// Device side of the inter-rack exchange described in megamoe_net.hpp. Nothing here depends on the routed GEMM
// specialization, so every helper takes explicit dimensions and buffers.
namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int NetSendThreads = 512;
constexpr int64_t NetSpinLimit = 1000000000;

/// Wait until `peer` has sent its message (combine = false) or results (combine = true) of launch `epoch`. The
/// pair's single semaphore counts both in strict alternation, so waiting does not consume a signal and any number of
/// threads may wait.
MSCCLPP_DEVICE_INLINE void netWait(const NetDevice& net, int peer, bool combine, uint64_t epoch) {
  uint64_t* inbound = net.channels[peer].semaphore_.inboundToken;
  const uint64_t expected = 2 * epoch - (combine ? 0 : 1);
  POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(inbound, memoryOrderAcquire) < expected), NetSpinLimit);
}

MSCCLPP_DEVICE_INLINE const NetHeader* netRecvHeader(const NetDevice& net, int index) {
  return reinterpret_cast<const NetHeader*>(net.base + net.layout.recvMeta + size_t(index) * net.layout.metaBytes);
}

MSCCLPP_DEVICE_INLINE const NetEntry* netRecvEntries(const NetDevice& net, int index) {
  return reinterpret_cast<const NetEntry*>(netRecvHeader(net, index) + 1);
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

// One CTA per network owner: pack and send this rank's message to that owner. Launched on the forward stream after
// input quantization and before the routed kernel, every launch, including empty ones.
template <int Threads = NetSendThreads>
__global__ void __launch_bounds__(Threads) netSendKernel(NetDevice net, NetSendArguments a) {
  __shared__ int sums[Threads / 32];
  __shared__ int tileTokens[Threads];
  const int index = blockIdx.x;
  const int owner = net.rankOf(index);
  const int nRoutes = a.nTokens * a.topK;
  auto ownerOf = [&](int id) {
    int rank = id >= 0 ? id / a.localExperts : -1;
    return rank < net.worldSize ? rank : -1;
  };
  uint8_t* message = net.base + net.layout.sendMeta + size_t(index) * net.layout.metaBytes;
  auto* header = reinterpret_cast<NetHeader*>(message);
  auto* entries = reinterpret_cast<NetEntry*>(header + 1);
  uint8_t* rows = net.base + net.layout.sendRows + size_t(index) * a.maxTokens * net.layout.rowBytes;
  auto* routeRows = reinterpret_cast<int32_t*>(net.base + net.layout.routeRows);

  // Owners' results are packed into combineIn in network-index order.
  int below = 0;
  for (int route = threadIdx.x; route < nRoutes; route += Threads) {
    int rank = ownerOf(a.ids[route]);
    bool netOwned = rank >= 0 && net.isNet(rank);
    if (netOwned && net.indexOf(rank) < index) ++below;
    if (index == 0 && !netOwned) routeRows[route] = -1;
  }
  int outBase;
  (void)netBlockExclusiveScan<Threads>(below, outBase, sums);

  constexpr int VectorBytes = 16;
  const int inputVectors = a.hidden / VectorBytes;
  const int rowVectors = inputVectors + a.scaleStride / VectorBytes;
  int nEntries = 0;
  int nRows = 0;
  for (int first = 0; first < a.nTokens; first += Threads) {
    const int token = first + threadIdx.x;
    int owned = 0;
    if (token < a.nTokens) {
      for (int slot = 0; slot < a.topK; ++slot) owned += ownerOf(a.ids[token * a.topK + slot]) == owner;
    }
    int tileEntries;
    int tileRows;
    int entry = nEntries + netBlockExclusiveScan<Threads>(owned, tileEntries, sums);
    int row = netBlockExclusiveScan<Threads>(owned ? 1 : 0, tileRows, sums);
    if (owned) {
      for (int slot = 0; slot < a.topK; ++slot) {
        int route = token * a.topK + slot;
        int id = a.ids[route];
        if (ownerOf(id) != owner) continue;
        entries[entry] = NetEntry{route, id - owner * a.localExperts, a.weights[route], nRows + row};
        routeRows[route] = outBase + entry;
        ++entry;
      }
      tileTokens[row] = token;
    }
    __syncthreads();
    for (int item = threadIdx.x; item < tileRows * rowVectors; item += Threads) {
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
    nEntries += tileEntries;
    nRows += tileRows;
    // The next tile reuses tileTokens.
    __syncthreads();
  }
  if (threadIdx.x == 0) *header = NetHeader{a.nTokens, nEntries, outBase, nRows};
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    PortChannelDeviceHandle channel = net.channels[owner];
    const int sourceIndex = net.indexFrom(owner, a.rank);
    channel.put(net.layout.recvMeta + size_t(sourceIndex) * net.layout.metaBytes,
                net.layout.sendMeta + size_t(index) * net.layout.metaBytes,
                sizeof(NetHeader) + size_t(nEntries) * sizeof(NetEntry));
    if (nRows)
      channel.put(net.layout.recvRows + size_t(sourceIndex) * a.maxTokens * net.layout.rowBytes,
                  net.layout.sendRows + size_t(index) * a.maxTokens * net.layout.rowBytes,
                  size_t(nRows) * net.layout.rowBytes);
    channel.signal();
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

/// Called by every thread of a CTA. Waits for the message of network source `index` and republishes its routes as
/// the current-epoch LL8 token count, chunk headers, and route packets (local expert << 16 | offset in its chunk of
/// RouteChunk routes) that an NVLink source writes remotely. Also records each route's entry for dispatch and FC2.
template <int RouteChunk>
MSCCLPP_DEVICE_INLINE void netRelaySource(const NetDevice& net, const NetRouting& r, int index, uint64_t epoch) {
  const int source = net.rankOf(index);
  const size_t routesPerRank = size_t(r.maxTokens) * r.topK;
  if (threadIdx.x == 0) netWait(net, source, false, epoch);
  __syncthreads();
  const NetHeader* header = netRecvHeader(net, index);
  const NetEntry* entries = netRecvEntries(net, index);
  const int nTokens = __ldcg(&header->nTokens);
  const int nEntries = __ldcg(&header->nEntries);
  if (r.tokenCount && threadIdx.x == 0) r.tokenCount[source].write(uint32_t(nTokens), uint32_t(epoch));
  int32_t* routeMap = netRouteMap(net, index, routesPerRank);
  mscclpp::LL8Packet* packets = r.packets + size_t(source) * routesPerRank;
  for (int entry = threadIdx.x; entry < nEntries; entry += blockDim.x) {
    int route = __ldcg(&entries[entry].route);
    int expert = __ldcg(&entries[entry].expert);
    int chunk = route / RouteChunk;
    int position = entry - netLowerBound(entries, nEntries, chunk * RouteChunk);
    routeMap[route] = entry;
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
}

/// Called by thread `index` < nNet of one CTA after every FC2 store of this launch has completed grid-wide: return the
/// staged result rows of network source `index` and signal it.
MSCCLPP_DEVICE_INLINE void netReturnResults(const NetDevice& net, int index, uint64_t epoch) {
  const int source = net.rankOf(index);
  // Routing skips the relay when every rank is empty, so the message may not have been observed yet.
  netWait(net, source, false, epoch);
  const NetHeader* header = netRecvHeader(net, index);
  const int nEntries = __ldcg(&header->nEntries);
  const int outBase = __ldcg(&header->outBase);
  __threadfence_system();
  PortChannelDeviceHandle channel = net.channels[net.worldSize + source];
  // The channel's source memory is this source's combineOut slot.
  if (nEntries)
    channel.put(net.layout.combineIn + size_t(outBase) * net.layout.outBytes, 0,
                size_t(nEntries) * net.layout.outBytes);
  channel.signal();
}

/// Called by every thread of every CTA: wait for every network owner's results, then copy this CTA's share of them
/// into the partial output rows [route][outBytes]. The caller synchronizes the grid before reading the partials.
MSCCLPP_DEVICE_INLINE void netCollectResults(const NetDevice& net, uint8_t* partialOutput, int nRoutes,
                                             uint64_t epoch) {
  for (int index = threadIdx.x; index < net.nNet; index += blockDim.x) netWait(net, net.rankOf(index), true, epoch);
  __syncthreads();
  const auto* routeRows = reinterpret_cast<const int32_t*>(net.base + net.layout.routeRows);
  const uint8_t* combineIn = net.base + net.layout.combineIn;
  const int lane = threadIdx.x % 32;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int vectors = int(net.layout.outBytes / sizeof(int4));
  for (int route = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; route < nRoutes; route += warps) {
    int row = routeRows[route];
    if (row < 0) continue;
    auto* source = reinterpret_cast<const int4*>(combineIn + size_t(row) * net.layout.outBytes);
    auto* destination = reinterpret_cast<int4*>(partialOutput + size_t(route) * net.layout.outBytes);
    for (int vector = lane; vector < vectors; vector += 32) destination[vector] = __ldcg(source + vector);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_NET_CUH_
