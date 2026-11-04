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

// Inter-rack (network) exchange.
//
// Ranks form contiguous racks of rackSize ranks; ranks of one rack share an NVLink domain and keep the direct CUDA IPC
// path. A rank outside the local rack is a network peer, reached through MSCCL++ PortChannels in device mode
// (GPUNetIO: GPU threads post the RDMA work requests and ring the NIC doorbell themselves, with no host proxy) on a
// registered network workspace. Network peers of a rank are numbered by their rank order with the local rack
// skipped: the network index of rank q seen from a rack starting at b is q < b ? q : q - rackSize.
//
// Per launch and ordered pair (source, owner) of network peers:
//  1. Before the routed kernel, netSendKernel packs the source's message: a NetHeader and the NetEntry of every route
//     owned by the owner in route order (the routes), and each token used by those routes once, as FP8 activations
//     followed by their E8M0 scales (the rows). It puts the routes into the owner's recvMeta slot and signals, then
//     puts the rows into the owner's recvRows slot and signals again.
//  2. In routing, the owner waits for the routes and republishes them as the current-epoch LL8 token count, route
//     headers, and route packets an NVLink source would have written, so the NVLink routing code is unchanged. It then
//     waits for the rows, whose transfer overlaps the relay.
//  3. Dispatch loads the token rows from recvRows, and FC2 stores each routed output row to the combineOut slot of its
//     source, in entry order.
//  4. After FC2, the owner returns the source's results to the source's combineIn rows starting at
//     NetHeader::outBase and signals the source. Without combineRouteWeights, FC2 rows are already weighted, so the
//     owner first sums the rows of each token row into reduceOut and returns one row per token row (NetDevice::reduce);
//     the source places it in the token's first route to that owner and zeros in its other routes to that owner.
//     Otherwise the owner returns the per-route combineOut rows. The source waits for every network owner and
//     scatters combineIn rows into its partial outputs before the NVLink join and combine.
// Each ordered pair shares one channel, so one queue pair and one semaphore. A rank's signals to a peer follow a fixed
// order: its launch-N routes (count 3N - 2), rows (3N - 1), then results (3N). Each signal is an RDMA atomic posted
// after the pair's writes on the same queue pair, so observing a count publishes the data it covers. Waits compare
// against these counts without consuming them. A source reuses its buffers only after every owner has returned the
// previous launch's results, which orders all buffer reuse without flushes. Once a source has its results it may send
// its next message, so after posting results an owner reads only local copies of the received header (recvInfo) and
// row boundaries (rowStart).
//
// Each queue pair is posted to by one thread at a time: part 0 of the owner's CTAs in netSendKernel, and the single
// warp completing a source's reduced rows (or one thread per source) in netReturnResults; the kernels are
// stream-ordered. Posts therefore use PortChannelSharingMode::Exclusive.
namespace mscclpp::megamoe {

constexpr int MaxWorldSize = 256;
constexpr int MaxRackSize = 72;

struct alignas(16) NetHeader {
  int32_t nTokens;   // source tokens in this launch
  int32_t nEntries;  // source routes owned by the destination
  int32_t outBase;   // first combineIn row of the destination's results on the source
  int32_t nRows;     // token rows sent to the destination
};

struct alignas(16) NetEntry {
  int32_t route;   // source route index, token * topK + slot
  int32_t expert;  // destination-local expert
  float weight;    // router weight
  int32_t row;     // token row in the destination's received rows
};

/// Device-mode (GPUNetIO) PortChannel handle used for every network peer.
using NetChannel = PortChannelDeviceHandleT<PortChannelBackend::Device>;

struct NetLayout {
  size_t bytes = 0;            // registered for IB as one memory region
  size_t slotBytes = 0;        // stride of the combineOut slots
  size_t metaBytes = 0;        // message stride: header and up to maxTokens * topK entries
  size_t rowBytes = 0;         // token row: FP8 activations, then E8M0 scales for MXFP8
  size_t outBytes = 0;         // partial output row
  size_t sendMeta = 0;         // [nNet] outgoing messages, by destination network index
  size_t recvMeta = 0;         // [nNet] incoming messages, by source network index
  size_t combineIn = 0;        // [maxTokens * topK] source-side result rows, grouped by owner network index
  size_t sendRows = 0;         // [nNet][maxTokens] outgoing token rows
  size_t recvRows = 0;         // [nNet][maxTokens] incoming token rows
  size_t combineOut = 0;       // [nNet] slots of [maxTokens * topK] owner-side result rows, by source network index
  size_t routeMap = 0;         // int [nNet][maxTokens * topK]: owner-side entry of each source route
  size_t routeRows = 0;        // int [maxTokens * topK]: source-side combineIn row of each route, -1, or NetZeroRow
  size_t reduceSlotBytes = 0;  // stride of the reduceOut slots
  size_t reduceOut = 0;        // [nNet] slots of [maxTokens] owner-side reduced result rows, by source network index
  size_t sendDone = 0;         // int [nNet]: send kernel parts done packing rows, by destination network index
  size_t returnDone = 0;       // int [nNet]: reduced rows done, by source network index
  size_t recvInfo = 0;         // NetHeader [nNet]: local copy of each received header, by source network index
  size_t rowStart = 0;  // int [nNet][maxTokens + 1]: first entry of each received token row, then the entry count
};

/// Host description of the network path, created by the runtime.
struct NetPlan {
  /// Ranks per NVLink rack; 0 or worldSize disables the network path.
  int rackSize = 0;
  /// Zero-initialized local network workspace of getNetLayout(config, rackSize).bytes, registered for IB.
  void* buffer = nullptr;
  /// Device array of worldSize device-mode PortChannel handles indexed by peer rank, from the local workspace to the
  /// peer's workspace. Only network peers are valid.
  const NetChannel* channels = nullptr;
};

/// Kernel view of the network path.
struct NetDevice {
  int rackSize = 1;
  int rackBegin = 0;
  int nNet = 0;
  int worldSize = 1;
  int maxTokens = 0;
  // Owners sum the weighted BF16 rows of each received token row and return one row per token row.
  bool reduce = false;
  NetLayout layout;
  uint8_t* base = nullptr;
  const NetChannel* channels = nullptr;

  MSCCLPP_HOST_DEVICE_INLINE bool isNet(int rank) const {
    return nNet && (rank < rackBegin || rank >= rackBegin + rackSize);
  }
  /// Network index of a rank outside the local rack.
  MSCCLPP_HOST_DEVICE_INLINE int indexOf(int rank) const { return rank < rackBegin ? rank : rank - rackSize; }
  /// Network index of rank `rank` seen from `viewer`.
  MSCCLPP_HOST_DEVICE_INLINE int indexFrom(int viewer, int rank) const {
    int begin = viewer / rackSize * rackSize;
    return rank < begin ? rank : rank - rackSize;
  }
  MSCCLPP_HOST_DEVICE_INLINE int rankOf(int index) const { return index < rackBegin ? index : index + rackSize; }
};

inline size_t netAligned(size_t n, size_t alignment = 256) { return (n + alignment - 1) / alignment * alignment; }

inline NetLayout getNetLayout(const NativeConfig& c, int rackSize) {
  NetLayout layout{};
  if (rackSize <= 0 || rackSize >= c.worldSize) return layout;
  const size_t nNet = size_t(c.worldSize - rackSize);
  const size_t routes = size_t(c.maxTokens) * c.topK;
  const size_t scaleStride = (size_t(c.hidden) / 32 + 15) / 16 * 16;
  const bool staticFp8 = c.activationDtype == ActivationDtype::Fp8Static;
  layout.metaBytes = netAligned(sizeof(NetHeader) + routes * sizeof(NetEntry));
  layout.rowBytes = size_t(c.hidden) + (staticFp8 ? 0 : scaleStride);
  layout.outBytes = size_t(c.hidden) * (staticFp8 ? 1 : 2);
  auto append = [&](size_t bytes) {
    size_t offset = netAligned(layout.bytes);
    layout.bytes = offset + bytes;
    return offset;
  };
  layout.sendMeta = append(nNet * layout.metaBytes);
  layout.recvMeta = append(nNet * layout.metaBytes);
  layout.combineIn = append(routes * layout.outBytes);
  layout.sendRows = append(nNet * c.maxTokens * layout.rowBytes);
  layout.recvRows = append(nNet * c.maxTokens * layout.rowBytes);
  layout.slotBytes = netAligned(routes * layout.outBytes);
  layout.combineOut = append(nNet * layout.slotBytes);
  layout.routeMap = append(nNet * routes * sizeof(int32_t));
  layout.routeRows = append(routes * sizeof(int32_t));
  layout.reduceSlotBytes = netAligned(size_t(c.maxTokens) * layout.outBytes);
  layout.reduceOut = append(nNet * layout.reduceSlotBytes);
  layout.sendDone = append(nNet * sizeof(int32_t));
  layout.returnDone = append(nNet * sizeof(int32_t));
  layout.recvInfo = append(nNet * sizeof(NetHeader));
  layout.rowStart = append(nNet * (size_t(c.maxTokens) + 1) * sizeof(int32_t));
  layout.bytes = netAligned(layout.bytes);
  return layout;
}

/// CTAs of netSendKernel per network owner.
inline int netSendParts(int nNet) { return nNet > 0 ? std::max(1, std::min(16, 128 / nNet)) : 1; }

inline NetDevice makeNetDevice(const NativeConfig& c, const NetPlan& plan) {
  NetDevice net{};
  net.worldSize = c.worldSize;
  net.maxTokens = c.maxTokens;
  net.rackSize = plan.rackSize > 0 ? plan.rackSize : c.worldSize;
  if (net.rackSize > c.worldSize || c.worldSize % net.rackSize)
    throw std::invalid_argument("MegaMoE rack size must divide worldSize");
  net.rackBegin = c.rank / net.rackSize * net.rackSize;
  net.nNet = c.worldSize - net.rackSize;
  if (!net.nNet) return net;
#if !defined(MSCCLPP_USE_GPUNETIO)
  throw std::invalid_argument(
      "MegaMoE inter-rack execution requires MSCCL++ built with the GPUNetIO backend (CUDA with InfiniBand)");
#endif
  if (!plan.buffer || !plan.channels)
    throw std::invalid_argument("MegaMoE inter-rack execution requires a network workspace and port channels");
  net.layout = getNetLayout(c, net.rackSize);
  // Rows are weighted on the owner only without combineRouteWeights; static FP8 partials are not summed.
  net.reduce = !c.combineRouteWeights && c.activationDtype != ActivationDtype::Fp8Static;
  net.base = static_cast<uint8_t*>(plan.buffer);
  net.channels = plan.channels;
  return net;
}

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_NET_HPP_
