// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "../config.hpp"
#include "../kernels/utils.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency_opt {

constexpr int kDispatchNWarps = 16;
constexpr int kDispatchMinNWarpsPerGroup = 8;
constexpr int kDispatchMaxNWarpGroups = kDispatchNWarps / kDispatchMinNWarpsPerGroup;
constexpr int kDispatchNThreads = kDispatchNWarps * WARP_SIZE;
constexpr int kDispatchMaxNSms = 128;
constexpr int kDispatchMaxNRecvTmaWorkers = kDispatchNWarps;
constexpr size_t kOptimizedDynamicSharedMemoryBytes = 226 * 1024;
constexpr size_t kTmaWorkerControlBytes = kDispatchMaxNWarpGroups * WARP_SIZE * sizeof(int);
static_assert(kDispatchNWarps % kDispatchMinNWarpsPerGroup == 0);
static_assert(sizeof(mscclpp::DeviceSemaphore) == sizeof(int));
static_assert(alignof(mscclpp::DeviceSemaphore) <= alignof(int));
static_assert(sizeof(mscclpp::DeviceSyncer) % sizeof(int) == 0);
static_assert(alignof(mscclpp::DeviceSyncer) <= alignof(int));

MSCCLPP_HOST_DEVICE_INLINE int rankSignalChannelIndex(int peerRank, int signalChannelStride) {
  return peerRank * signalChannelStride;
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataBytes(int nRanks, int nExperts) {
  return configAlign<size_t>(static_cast<size_t>(nRanks + nExperts) * sizeof(mscclpp::LL8Packet), 128);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchPayloadStride(int hidden, int nTopk) {
  return configAlign<size_t>(LowLatencyPayloadView<nv_bfloat16>(hidden, nTopk).numBytes_, 128);
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNWarpsPerGroup(int nTokens, int nBlocks) {
  return nTokens <= nBlocks ? kDispatchNWarps
                            : (nTokens <= 2 * nBlocks ? kDispatchNWarps / 2 : kDispatchMinNWarpsPerGroup);
}

struct RecvTask {
  int sourceRank_;
  int tokenBegin_;
  int tokenEnd_;
};

struct DispatchWorkspaceView {
  uint32_t* metadataEpoch_;
  int* rankPayloadSlots_;
  int* rankPayloadCompletions_;
  mscclpp::DeviceSemaphore* localPayloadReady_;
  int* recvExpertCopiedCounts_;
  uint32_t* rankReadyEpochs_;
  RecvTask* recvTasks_;
  uint32_t* tasksAssignedEpoch_;
  int* nRecvTasks_;
  mscclpp::DeviceSyncer* combineSyncer_;

  MSCCLPP_HOST_DEVICE_INLINE DispatchWorkspaceView(void* workspace, int nRanks, int nExperts,
                                                   [[maybe_unused]] int maxSms) {
    auto* cursor = reinterpret_cast<int*>(workspace);
    metadataEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    rankPayloadSlots_ = cursor;
    cursor += nRanks;
    rankPayloadCompletions_ = cursor;
    cursor += nRanks;
    localPayloadReady_ = reinterpret_cast<mscclpp::DeviceSemaphore*>(cursor++);
    recvExpertCopiedCounts_ = cursor;
    cursor += nExperts;
    rankReadyEpochs_ = reinterpret_cast<uint32_t*>(cursor);
    cursor += nRanks;
    recvTasks_ = reinterpret_cast<RecvTask*>(cursor);
    cursor += 3 * kDispatchMaxNSms;
    tasksAssignedEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    nRecvTasks_ = cursor++;
    combineSyncer_ = reinterpret_cast<mscclpp::DeviceSyncer*>(cursor);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int nRanks, int nExperts, [[maybe_unused]] int maxSms) {
    return static_cast<size_t>(3 * nRanks + nExperts + 3 * kDispatchMaxNSms + 4) * sizeof(int) +
           sizeof(mscclpp::DeviceSyncer);
  }
};

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchWorkspaceBytes(int nRanks, int nExperts, int maxSms) {
  return DispatchWorkspaceView::numBytes(nRanks, nExperts, maxSms);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedControlBytes(int nRanks) {
  constexpr int kNSendSlots = kDispatchMaxNWarpGroups * WARP_SIZE;
  const int nSlots = nRanks > kNSendSlots ? nRanks : kNSendSlots;
  return configAlign<size_t>(static_cast<size_t>(nSlots) * sizeof(int), 128);
}

template <int kHidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSendTmaBytes(int nTopk) {
  return kDispatchMaxNWarpGroups * (dispatchPayloadStride(kHidden, nTopk) + sizeof(uint64_t));
}

template <int kHidden, int kMaxWorkers>
MSCCLPP_HOST_DEVICE_INLINE constexpr int tmaWorkerCount() {
  static_assert(kHidden % 128 == 0);
  constexpr size_t workerBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16) + sizeof(uint64_t);
  constexpr int nWorkers =
      static_cast<int>((kOptimizedDynamicSharedMemoryBytes - kTmaWorkerControlBytes) / workerBytes);
  return nWorkers < kMaxWorkers ? nWorkers : kMaxWorkers;
}

template <int kHidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchRecvTmaBytes() {
  constexpr int kNWorkers = tmaWorkerCount<kHidden, kDispatchMaxNRecvTmaWorkers>();
  constexpr size_t tileBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  return static_cast<size_t>(kNWorkers) * (tileBytes + sizeof(uint64_t));
}

template <int kHidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedBytes(int nRanks, int nExperts, int nTopk) {
  const size_t controlBytes = dispatchSharedControlBytes(nRanks);
  const size_t sendBytes = dispatchSendTmaBytes<kHidden>(nTopk);
  const size_t recvBytes = dispatchRecvTmaBytes<kHidden>();
  const size_t tmaBytes = controlBytes + (sendBytes > recvBytes ? sendBytes : recvBytes);
  const size_t metadataBytes = static_cast<size_t>(nRanks + nExperts) * sizeof(int);
  return tmaBytes > metadataBytes ? tmaBytes : metadataBytes;
}

}  // namespace low_latency_opt
}  // namespace ep
}  // namespace mscclpp
