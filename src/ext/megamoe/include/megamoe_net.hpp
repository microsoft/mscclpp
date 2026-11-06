// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_NET_HPP_
#define MSCCLPP_EXT_MEGAMOE_NET_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mscclpp/device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <stdexcept>

#include "megamoe_kernel.hpp"

// Inter-rack (network) exchange, aggregated per rack over NIC rails.
//
// Ranks form contiguous racks of rackSize ranks that share an NVLink domain. Rank r has rack r / rackSize and lane
// r % rackSize. Every rank owns one NIC, and the NICs of equal lanes form a rail: a rank talks over IB only to its rail
// peers, the ranks of its lane in the other racks. The rail peers are numbered by rack with the local rack skipped
// (the rack's network index), and the rail peer of a remote rack is that rack's forwarder for this rank.
//
// Per launch and ordered pair (source s, remote rack B, whose forwarder f is s's rail peer in B):
//  1. Send: the input quantization also writes every token row (FP8 activations then E8M0 scales) to sendRows. At the
//     start of the routed kernel, one CTA of s puts all token rows to f and signals (Rows), while another builds a
//     message of the routes owned in B: a NetHeader, per-chunk route counts for every owner lane, and a NetEntry per
//     route that already carries its position among the chunk's entries for that owner. It puts the message to f and
//     signals (Routes).
//  2. Relay: in routing, a CTA of f writes the routes into every owner's LL8 token count, chunk headers, and route
//     packets of source s over NVLink, as if s shared the rack, and each route's weight into the owner's weight
//     packets. Owners then route s like any NVLink source.
//  3. Dispatch: a network agent warp of f, idle otherwise, waits for the rows and publishes their arrival to every
//     owner of the rack (rowsReady). An owner's dispatch waits only for the rows of the network source it copies, and
//     copies them from f's recvRows over NVLink; nothing waits for rows before the GEMM starts.
//  4. Combine: the FC2 epilogue of an owner adds each weighted output row of a route of s into f's accum row of the
//     token with a bulk BF16 reduction over NVLink, so f collects one rack-level partial sum per token while FC2 runs.
//     Once its last CTA's epilogue has drained, every owner signals each forwarder of its rack (combineDone). The CTA
//     of f serving s then puts the source's accum rows to s's combineIn and signals (Results). The final combine of s
//     skips its routes owned in other racks and adds the rack partials instead.
// So each token crosses IB at most once per remote rack in each direction.
//
// Each rail pair uses NetPlan::qps channels, each its own queue pair and semaphore. Routes travel on queue pair 0,
// rows on the other queue pairs (on queue pair 0 too if it is the only one), and results on every queue pair; rows and
// results are split by row, and every queue pair carrying a phase signals its share, even when empty. Each queue
// pair's signals per launch follow the order routes, rows, results, so a waiter compares the queue pair's count
// against the position of the phase among the phases it carries (netExpected) without consuming it. Each signal is an
// RDMA atomic posted after the writes on the same queue pair, so observing a count publishes the data it covers. A
// source reuses its buffers only after every forwarder returned the previous launch's results, which orders all buffer
// reuse without flushes. accum is double-buffered by launch parity: launch N zeroes the rows launch N - 2 used before
// any owner may add to them (step 3 is published after routing's grid barriers), and the source received launch
// N - 2's results before it sent launch N - 1's message, so the NIC no longer reads them.
//
// Each queue pair is posted to by one thread at a time: by the source's sending CTAs before routing completes, and by
// one warp per queue pair of the forwarder's serving CTA after combineDone. Posts therefore use
// PortChannelSharingMode::Exclusive.
namespace mscclpp::megamoe {

constexpr int MaxWorldSize = 256;
constexpr int MaxRackSize = 72;
/// Queue pairs per rail peer: default and maximum (MSCCLPP_MEGAMOE_NET_QPS overrides the default).
constexpr int DefaultNetQps = 4;
constexpr int MaxNetQps = 8;
/// Smallest routing chunk of any kernel specialization; bounds the per-chunk count tables.
constexpr int NetMinRouteChunk = 384;
/// Smallest share of a row transfer worth its own queue pair.
constexpr uint64_t NetMinQpBytes = 64 * 1024;

struct alignas(16) NetHeader {
  int32_t nTokens;   // source tokens in this launch
  int32_t nEntries;  // source routes owned in the destination rack
  int32_t nChunks;   // source routing chunks; counts holds nChunks * rackSize entries
  int32_t reserved;
};

struct alignas(16) NetEntry {
  int32_t route;      // source route index, token * topK + slot
  uint16_t expert;    // owner-local expert
  uint16_t position;  // index among the entries of this route's chunk owned by the same rank
  float weight;       // router weight
  int32_t owner;      // owner lane in the destination rack
};

/// Device-mode (GPUNetIO) PortChannel handle used for every rail peer.
using NetChannel = PortChannelDeviceHandleT<PortChannelBackend::Device>;

struct NetLayout {
  size_t bytes = 0;         // registered for IB and CUDA IPC as one memory region
  size_t metaBytes = 0;     // message stride: header, counts, and up to maxTokens * topK entries
  size_t countsBytes = 0;   // per-chunk owner counts of a message, padded to 16 bytes
  size_t rowBytes = 0;      // token row: FP8 activations, then E8M0 scales
  size_t outBytes = 0;      // BF16 output row
  size_t sendMeta = 0;      // [nNet] outgoing messages, by destination rack network index
  size_t recvMeta = 0;      // [nNet] incoming messages, by source rack network index
  size_t sendRows = 0;      // [maxTokens] this rank's token rows
  size_t recvRows = 0;      // [nNet][maxTokens] rail peers' token rows, read by the rack's owners
  size_t accum = 0;         // [2][nNet][maxTokens] BF16 rack partial sums for rail peers, by launch parity
  size_t combineIn = 0;     // [nNet][maxTokens] BF16 partial sums of the remote racks for this rank's tokens
  size_t weights = 0;       // LL8 [worldSize - rackSize][maxTokens * topK] weights of network sources' routes
  size_t rowsReady = 0;     // uint64 [worldSize - rackSize]: epoch whose rows of each network source are readable
  size_t accumTokens = 0;   // int [2][nNet]: accum rows each parity used, zeroed at its next use
  size_t combineDone = 0;   // uint64: rack owners whose epilogue drained, over all launches
  size_t epilogueDone = 0;  // uint32: CTAs whose epilogue drained in this launch
};

/// Host description of the network path, created by the runtime.
struct NetPlan {
  /// Ranks per NVLink rack; 0 or worldSize disables the network path.
  int rackSize = 0;
  /// Zero-initialized local network workspace of getNetLayout(config, rackSize).bytes, registered for IB and CUDA IPC.
  void* buffer = nullptr;
  /// Device array of qps * nNet device-mode PortChannel handles indexed by queue pair, then remote rack network index,
  /// from the local workspace to the rail peer's workspace.
  const NetChannel* channels = nullptr;
  /// Queue pairs (channels) per rail peer.
  int qps = 1;
  /// Device array of rackSize network workspace addresses of the local rack's ranks, by lane (CUDA IPC mappings).
  const uint64_t* rackBases = nullptr;
};

/// Kernel view of the network path.
struct NetDevice {
  int rackSize = 1;
  int rackBegin = 0;
  int rack = 0;
  int lane = 0;
  int nNet = 0;
  int worldSize = 1;
  int maxTokens = 0;
  int qps = 1;
  NetLayout layout;
  uint8_t* base = nullptr;
  const uint64_t* rackBases = nullptr;
  const NetChannel* channels = nullptr;

  /// Channel of queue pair `qp` to the rail peer in the remote rack of network index `index`.
  MSCCLPP_HOST_DEVICE_INLINE const NetChannel& channel(int qp, int index) const {
    return channels[size_t(qp) * nNet + index];
  }
  MSCCLPP_HOST_DEVICE_INLINE bool isNet(int rank) const {
    return nNet && (rank < rackBegin || rank >= rackBegin + rackSize);
  }
  /// Index of a rank outside the local rack among those ranks, in rank order.
  MSCCLPP_HOST_DEVICE_INLINE int sourceIndex(int rank) const { return rank < rackBegin ? rank : rank - rackSize; }
  /// Network index of a remote rack; it is the same for every rank of the local rack.
  MSCCLPP_HOST_DEVICE_INLINE int rackIndex(int remoteRack) const {
    return remoteRack < rack ? remoteRack : remoteRack - 1;
  }
  MSCCLPP_HOST_DEVICE_INLINE int rackOfIndex(int index) const { return index < rack ? index : index + 1; }
  /// Rail peer in the remote rack of network index `index`.
  MSCCLPP_HOST_DEVICE_INLINE int railPeer(int index) const { return rackOfIndex(index) * rackSize + lane; }
  /// Network index of this rack as seen from the remote rack of network index `index`.
  MSCCLPP_HOST_DEVICE_INLINE int indexAt(int index) const { return rack < rackOfIndex(index) ? rack : rack - 1; }
  MSCCLPP_HOST_DEVICE_INLINE uint8_t* rackBase(int peerLane) const {
    return reinterpret_cast<uint8_t*>(rackBases[peerLane]);
  }
  /// The forwarder of network source `source` in this rack, which holds its rows and accumulates its results.
  MSCCLPP_HOST_DEVICE_INLINE uint8_t* forwarderBase(int source) const { return rackBase(source % rackSize); }
};

inline size_t netAligned(size_t n, size_t alignment = 256) { return (n + alignment - 1) / alignment * alignment; }

inline NetLayout getNetLayout(const NativeConfig& c, int rackSize) {
  NetLayout layout{};
  if (rackSize <= 0 || rackSize >= c.worldSize) return layout;
  const size_t nNet = size_t(c.worldSize / rackSize - 1);
  const size_t routes = size_t(c.maxTokens) * c.topK;
  const size_t chunks = (routes + NetMinRouteChunk - 1) / NetMinRouteChunk;
  const size_t scaleStride = (size_t(c.hidden) / 32 + 15) / 16 * 16;
  layout.countsBytes = netAligned(chunks * rackSize * sizeof(int32_t), 16);
  layout.metaBytes = netAligned(sizeof(NetHeader) + layout.countsBytes + routes * sizeof(NetEntry));
  layout.rowBytes = size_t(c.hidden) + scaleStride;
  layout.outBytes = size_t(c.hidden) * 2;
  auto append = [&](size_t bytes) {
    size_t offset = netAligned(layout.bytes);
    layout.bytes = offset + bytes;
    return offset;
  };
  layout.sendMeta = append(nNet * layout.metaBytes);
  layout.recvMeta = append(nNet * layout.metaBytes);
  layout.sendRows = append(c.maxTokens * layout.rowBytes);
  layout.recvRows = append(nNet * c.maxTokens * layout.rowBytes);
  layout.accum = append(2 * nNet * c.maxTokens * layout.outBytes);
  layout.combineIn = append(nNet * c.maxTokens * layout.outBytes);
  layout.weights = append(size_t(c.worldSize - rackSize) * routes * 8);
  layout.rowsReady = append(size_t(c.worldSize - rackSize) * sizeof(uint64_t));
  layout.accumTokens = append(2 * nNet * sizeof(int32_t));
  layout.combineDone = append(sizeof(uint64_t));
  layout.epilogueDone = append(sizeof(uint32_t));
  layout.bytes = netAligned(layout.bytes);
  return layout;
}

inline NetDevice makeNetDevice(const NativeConfig& c, const NetPlan& plan) {
  NetDevice net{};
  net.worldSize = c.worldSize;
  net.maxTokens = c.maxTokens;
  net.rackSize = plan.rackSize > 0 ? plan.rackSize : c.worldSize;
  if (net.rackSize > c.worldSize || c.worldSize % net.rackSize)
    throw std::invalid_argument("MegaMoE rack size must divide worldSize");
  net.rack = c.rank / net.rackSize;
  net.lane = c.rank % net.rackSize;
  net.rackBegin = net.rack * net.rackSize;
  net.nNet = c.worldSize / net.rackSize - 1;
  if (!net.nNet) return net;
#if !defined(MSCCLPP_USE_GPUNETIO)
  throw std::invalid_argument(
      "MegaMoE inter-rack execution requires MSCCL++ built with the GPUNetIO backend (CUDA with InfiniBand)");
#endif
  if (!plan.buffer || !plan.channels || !plan.rackBases)
    throw std::invalid_argument("MegaMoE inter-rack execution requires a network workspace and rail channels");
  // Racks sum BF16 partials; static FP8 partials are not summable.
  if (c.activationDtype != ActivationDtype::Mxfp8)
    throw std::invalid_argument("MegaMoE inter-rack execution supports only MXFP8 activations");
  if (plan.qps < 1 || plan.qps > MaxNetQps)
    throw std::invalid_argument("MegaMoE network queue pairs must be in [1, 8]");
  net.layout = getNetLayout(c, net.rackSize);
  net.base = static_cast<uint8_t*>(plan.buffer);
  net.channels = plan.channels;
  net.rackBases = plan.rackBases;
  net.qps = plan.qps;
  return net;
}

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_NET_HPP_
