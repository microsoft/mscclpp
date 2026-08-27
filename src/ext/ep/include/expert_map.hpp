// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Single source of truth for the expert -> rank mapping used by every EP
// dispatch and combine algorithm.
//
// Without ETP (etpSize == 1) every helper below reduces algebraically to the
// expressions the kernels used to open-code:
//
//   nLocalExperts = numExperts / numRanks
//   dstRank       = expert / nLocalExperts
//
// With etpSize > 1 an expert is owned by an EP *group* of etpSize ranks that
// tensor-shard its weights, so "the owner" becomes a group index plus a
// per-sender leader inside that group.

#ifndef MSCCLPP_EP_EXPERT_MAP_HPP_
#define MSCCLPP_EP_EXPERT_MAP_HPP_

#include <mscclpp/device.hpp>
#include <mscclpp/ext/ep/types.hpp>

#include "device_context.hpp"

namespace mscclpp {
namespace ep {

/// Expert placement and dispatch routing for one rank.
struct ExpertMap {
  MoETopology topology_;
  int numExperts_;
  int numExpertsPerGroup_;
  int rank_;
  int epIndex_;
  int tpIndex_;
  EtpDispatchMode dispatchMode_;

  MSCCLPP_HOST_DEVICE_INLINE ExpertMap(const MoETopology& topology, int numExperts, int rank,
                                       EtpDispatchMode dispatchMode)
      : topology_(topology),
        numExperts_(numExperts),
        numExpertsPerGroup_(topology.numExpertsPerGroup(numExperts)),
        rank_(rank),
        epIndex_(topology.epIndex(rank)),
        tpIndex_(topology.tpIndex(rank)),
        dispatchMode_(dispatchMode) {}

  MSCCLPP_HOST_DEVICE_INLINE ExpertMap(const DeviceContext* context, int numExperts)
      : ExpertMap(context->topology_, numExperts, context->rank_, context->etpDispatchMode_) {}

  MSCCLPP_HOST_DEVICE_INLINE int numRanks() const { return topology_.numRanks; }
  MSCCLPP_HOST_DEVICE_INLINE int epSize() const { return topology_.epSize; }
  MSCCLPP_HOST_DEVICE_INLINE int etpSize() const { return topology_.etpSize; }
  MSCCLPP_HOST_DEVICE_INLINE bool isEtpEnabled() const { return topology_.isEtpEnabled(); }
  MSCCLPP_HOST_DEVICE_INLINE bool duplicateSend() const {
    return dispatchMode_ == EtpDispatchMode::DUPLICATE_SEND;
  }

  /// Number of experts owned by one EP group; equals numExperts / numRanks
  /// when etpSize == 1.
  MSCCLPP_HOST_DEVICE_INLINE int numLocalExperts() const { return numExpertsPerGroup_; }
  /// EP group owning @p expert, or -1 when @p expert is invalid.
  MSCCLPP_HOST_DEVICE_INLINE int group(int expert) const { return topology_.expertGroup(expert, numExpertsPerGroup_); }
  /// Group-local index of @p expert, or -1 when @p expert is invalid.
  MSCCLPP_HOST_DEVICE_INLINE int localExpert(int expert) const {
    return topology_.localExpert(expert, numExpertsPerGroup_);
  }
  /// First global expert index owned by this rank's group.
  MSCCLPP_HOST_DEVICE_INLINE int globalExpertBase() const { return epIndex_ * numExpertsPerGroup_; }
  /// Whether this rank holds a shard of @p expert.
  MSCCLPP_HOST_DEVICE_INLINE bool ownsExpert(int expert) const {
    return expert >= 0 && group(expert) == epIndex_;
  }
  /// Whether @p rank holds a shard of @p expert.
  MSCCLPP_HOST_DEVICE_INLINE bool rankOwnsExpert(int rank, int expert) const {
    return expert >= 0 && group(expert) == topology_.epIndex(rank);
  }

  /// Destination rank this rank sends a token routed to @p expert to.
  ///
  /// LEADER_SINGLE_SEND picks the group member with the sender's own tpIndex,
  /// so the all-to-all degenerates into an epSize-sized permutation and every
  /// source rank has a distinct leader. With etpSize == 1 this is the owner.
  MSCCLPP_HOST_DEVICE_INLINE int leaderRank(int expert) const {
    return topology_.leaderRank(expert, numExpertsPerGroup_, tpIndex_);
  }
  /// Number of copies of a routed token this rank sends per destination group.
  MSCCLPP_HOST_DEVICE_INLINE int numSendCopies() const { return duplicateSend() ? etpSize() : 1; }
  /// Destination rank of send copy @p copyIdx for a token routed to @p expert.
  MSCCLPP_HOST_DEVICE_INLINE int destinationRank(int expert, int copyIdx) const {
    const int expertGroup = group(expert);
    if (expertGroup < 0) return -1;
    return duplicateSend() ? topology_.rankOf(expertGroup, copyIdx) : topology_.rankOf(expertGroup, tpIndex_);
  }
  /// Rank whose receive buffer holds @p sourceRank's payloads for this rank.
  ///
  /// This is the fused-pull peer of design B2: with LEADER_SINGLE_SEND only the
  /// group member sharing @p sourceRank's tpIndex received the row. With
  /// etpSize == 1, or with DUPLICATE_SEND, this is always the local rank.
  MSCCLPP_HOST_DEVICE_INLINE int sourcePeer(int sourceRank) const {
    if (duplicateSend() || !isEtpEnabled()) return rank_;
    return topology_.rankOf(epIndex_, topology_.tpIndex(sourceRank));
  }
  /// Rank in @p destinationRank's group that actually receives this rank's
  /// payloads; used to convert a signal target into a payload-count bucket.
  MSCCLPP_HOST_DEVICE_INLINE int leaderInGroupOf(int destinationRank) const {
    if (duplicateSend() || !isEtpEnabled()) return destinationRank;
    return topology_.rankOf(topology_.epIndex(destinationRank), tpIndex_);
  }

  /// Number of per-(source, group-local expert) count slots in the dispatch
  /// metadata header. Equals numExperts when etpSize == 1.
  MSCCLPP_HOST_DEVICE_INLINE int metadataExpertSlots() const {
    return (duplicateSend() ? numRanks() : epSize()) * numExpertsPerGroup_;
  }
  /// Metadata slot (relative to the expert-count region) written by
  /// @p sourceRank for group-local expert @p localExpertIdx.
  MSCCLPP_HOST_DEVICE_INLINE int metadataExpertSlot(int sourceRank, int localExpertIdx) const {
    const int sourceSlot = duplicateSend() ? sourceRank : topology_.epIndex(sourceRank);
    return sourceSlot * numExpertsPerGroup_ + localExpertIdx;
  }
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_EXPERT_MAP_HPP_
