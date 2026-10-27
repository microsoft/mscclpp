// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_NET_HPP_
#define MSCCLPP_EXT_MEGAMOE_NET_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mscclpp/device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <stdexcept>
#include <string>

#include "megamoe_kernel.hpp"

// Inter-rack (network) exchange.
//
// Ranks form contiguous racks of rackSize ranks; ranks of one rack share an NVLink domain and keep the direct CUDA IPC
// path. A rank outside the local rack is a network peer, reached through MSCCL++ PortChannels (CPU proxy, RDMA over
// IB) on a registered network workspace. Network peers of a rank are numbered by their rank order with the local rack
// skipped: the network index of rank q seen from a rack starting at b is q < b ? q : q - rackSize.
//
// Per launch and ordered pair (source, owner) of network peers:
//  1. Before the routed kernel, the source packs a message: a NetHeader and the NetEntry of every route owned by the
//     owner in route order, plus each token used by those routes once, as FP8 activations followed by their E8M0
//     scales. It puts the message into the owner's recvMeta/recvRows slot and signals the owner.
//  2. In routing, the owner waits for that signal and republishes the entries as the current-epoch LL8 token count,
//     route headers, and route packets an NVLink source would have written, so the NVLink routing code is unchanged.
//  3. Dispatch loads the token rows from recvRows, and FC2 stores each routed output row to the combineOut slot of its
//     source, in entry order.
//  4. After FC2, the owner puts the source's slot to the source's combineIn rows starting at NetHeader::outBase and
//     signals the source. The source waits for every network owner and scatters combineIn rows into its
//     partial outputs before the NVLink join and combine.
// Each ordered pair shares one semaphore. A rank's signals to a peer strictly alternate through one proxy FIFO and
// queue pair: its launch-N message (count 2N - 1), then its launch-N results (count 2N). Waits compare against these
// counts without consuming them. A source reuses its buffers only after every owner has returned the previous
// launch's rows, which orders all buffer reuse without extra flushes.
//
// Port-channel triggers carry 32-bit offsets, so every put endpoint except the owner-side result slots lies in a
// leading main region; each result slot is registered as its own memory and put from offset 0.
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

struct NetLayout {
  size_t bytes = 0;
  size_t mainBytes = 0;   // registered region holding every put endpoint except combineOut
  size_t slotBytes = 0;   // stride of the separately registered combineOut slots
  size_t metaBytes = 0;   // message stride: header and up to maxTokens * topK entries
  size_t rowBytes = 0;    // token row: FP8 activations, then E8M0 scales for MXFP8
  size_t outBytes = 0;    // partial output row
  size_t sendMeta = 0;    // [nNet] outgoing messages, by destination network index
  size_t recvMeta = 0;    // [nNet] incoming messages, by source network index
  size_t combineIn = 0;   // [maxTokens * topK] source-side result rows, grouped by owner network index
  size_t sendRows = 0;    // [nNet][maxTokens] outgoing token rows
  size_t recvRows = 0;    // [nNet][maxTokens] incoming token rows
  size_t combineOut = 0;  // [nNet] slots of [maxTokens * topK] owner-side result rows, by source network index
  size_t routeMap = 0;    // int [nNet][maxTokens * topK]: owner-side entry of each source route
  size_t routeRows = 0;   // int [maxTokens * topK]: source-side combineIn row of each route, or -1
};

/// Host description of the network path, created by the runtime.
struct NetPlan {
  /// Ranks per NVLink rack; 0 or worldSize disables the network path.
  int rackSize = 0;
  /// Zero-initialized local network workspace of getNetLayout(config, rackSize).bytes, registered for IB.
  void* buffer = nullptr;
  /// Device array of 2 * worldSize PortChannel handles indexed by peer rank, both on the peer's semaphore and with the
  /// peer's main region as destination: first with the local main region as source, then with the combineOut slot of
  /// the peer as source. Only network peers are valid.
  const PortChannelDeviceHandle* channels = nullptr;
};

/// Kernel view of the network path.
struct NetDevice {
  int rackSize = 1;
  int rackBegin = 0;
  int nNet = 0;
  int worldSize = 1;
  NetLayout layout;
  uint8_t* base = nullptr;
  const PortChannelDeviceHandle* channels = nullptr;

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
  layout.mainBytes = netAligned(layout.bytes);
  layout.slotBytes = netAligned(routes * layout.outBytes);
  layout.combineOut = append(nNet * layout.slotBytes);
  layout.routeMap = append(nNet * routes * sizeof(int32_t));
  layout.routeRows = append(routes * sizeof(int32_t));
  layout.bytes = netAligned(layout.bytes);
  constexpr size_t TriggerLimit = size_t(std::numeric_limits<uint32_t>::max());
  if (layout.mainBytes > TriggerLimit || layout.slotBytes > TriggerLimit)
    throw std::invalid_argument("MegaMoE inter-rack transfers exceed 32-bit port-channel offsets (" +
                                std::to_string(layout.mainBytes) +
                                " registered bytes); reduce maxTokens or use larger racks");
  return layout;
}

inline NetDevice makeNetDevice(const NativeConfig& c, const NetPlan& plan) {
  NetDevice net{};
  net.worldSize = c.worldSize;
  net.rackSize = plan.rackSize > 0 ? plan.rackSize : c.worldSize;
  if (net.rackSize > c.worldSize || c.worldSize % net.rackSize)
    throw std::invalid_argument("MegaMoE rack size must divide worldSize");
  net.rackBegin = c.rank / net.rackSize * net.rackSize;
  net.nNet = c.worldSize - net.rackSize;
  if (!net.nNet) return net;
  if (!plan.buffer || !plan.channels)
    throw std::invalid_argument("MegaMoE inter-rack execution requires a network workspace and port channels");
  net.layout = getNetLayout(c, net.rackSize);
  net.base = static_cast<uint8_t*>(plan.buffer);
  net.channels = plan.channels;
  return net;
}

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_NET_HPP_
