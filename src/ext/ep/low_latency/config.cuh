// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "../config.hpp"
#include "api.cuh"
#include "device_helpers.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency_opt {

constexpr int DispatchNWarps = 16;
constexpr int DispatchMinNWarpsPerGroup = 8;
constexpr int DispatchMaxNWarpGroups = DispatchNWarps / DispatchMinNWarpsPerGroup;
constexpr int DispatchNThreads = DispatchNWarps * WARP_SIZE;
constexpr int DispatchMaxNRecvTmaWorkers = DispatchNWarps;
constexpr size_t OptimizedDynamicSharedMemoryBytes = 226 * 1024;
constexpr size_t TmaWorkerControlBytes = DispatchMaxNWarpGroups * WARP_SIZE * sizeof(int);
static_assert(DispatchNWarps % DispatchMinNWarpsPerGroup == 0);
static_assert(sizeof(mscclpp::DeviceSyncer) % sizeof(int) == 0);
static_assert(alignof(mscclpp::DeviceSyncer) <= alignof(int));

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataBytes(int nRanks, int nExperts) {
  return configAlign<size_t>(static_cast<size_t>(2 * nRanks + nExperts) * sizeof(mscclpp::LL8Packet), 128);
}

MSCCLPP_DEVICE_INLINE mscclpp::LL8Packet* dispatchReadyPackets(void* recvBuffer, int nRanks, int nExperts) {
  return reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks + nExperts;
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchPayloadStride(int hidden, int nTopk) {
  return configAlign<size_t>(low_latency::PayloadView<__bfloat16>(hidden, nTopk).numBytes_, 128);
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNWarpsPerGroup(int nTokens, int nBlocks) {
  return nTokens <= nBlocks ? DispatchNWarps
                            : (nTokens <= 2 * nBlocks ? DispatchNWarps / 2 : DispatchMinNWarpsPerGroup);
}

struct RecvTask {
  int sourceRank_;
  int tokenBegin_;
  int tokenEnd_;
};

struct WorkspaceView {
  uint32_t* metadataEpoch_;
  int* rankPayloadSlots_;
  int* rankPayloadCompletions_;
  int* recvExpertCopiedCounts_;
  uint32_t* rankReadyEpochs_;
  RecvTask* recvTasks_;
  uint32_t* tasksAssignedEpoch_;
  int* nRecvTasks_;
  mscclpp::DeviceSyncer* combineSyncer_;

  MSCCLPP_HOST_DEVICE_INLINE WorkspaceView(void* workspace, int nRanks, int nExperts) {
    auto* cursor = reinterpret_cast<int*>(workspace);
    metadataEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    rankPayloadSlots_ = cursor;
    cursor += nRanks;
    rankPayloadCompletions_ = cursor;
    cursor += nRanks;
    recvExpertCopiedCounts_ = cursor;
    cursor += nExperts;
    rankReadyEpochs_ = reinterpret_cast<uint32_t*>(cursor);
    cursor += nRanks;
    recvTasks_ = reinterpret_cast<RecvTask*>(cursor);
    cursor += 3 * low_latency::MaxWorkerBlocks;
    tasksAssignedEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    nRecvTasks_ = cursor++;
    combineSyncer_ = reinterpret_cast<mscclpp::DeviceSyncer*>(cursor);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int nRanks, int nExperts) {
    return static_cast<size_t>(3 * nRanks + nExperts + 3 * low_latency::MaxWorkerBlocks + 3) * sizeof(int) +
           sizeof(mscclpp::DeviceSyncer);
  }
};

struct KernelConfigCache {
  int deviceId_ = -1;
  size_t dynamicSharedBytes_ = 0;
  int residentBlocks_ = 0;
};

template <typename Kernel>
inline int configureKernel(Kernel kernel, int nThreads, size_t dynamicSharedBytes, const low_latency::CommContext& comm,
                           KernelConfigCache& cache) {
  if (cache.deviceId_ != comm.deviceId_ || cache.dynamicSharedBytes_ < dynamicSharedBytes) {
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    EP_HOST_ASSERT(dynamicSharedBytes + attributes.sharedSizeBytes <=
                   static_cast<size_t>(comm.maxSharedMemoryPerBlock_));
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamicSharedBytes)));
    int blocksPerSm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, nThreads, dynamicSharedBytes));
    cache.deviceId_ = comm.deviceId_;
    cache.dynamicSharedBytes_ = dynamicSharedBytes;
    cache.residentBlocks_ = blocksPerSm * comm.numSms_;
  }
  return cache.residentBlocks_;
}

MSCCLPP_HOST_DEVICE_INLINE size_t workspaceBytes(int nRanks, int nExperts) {
  return WorkspaceView::numBytes(nRanks, nExperts);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedControlBytes(int nRanks) {
  constexpr int NSendSlots = DispatchMaxNWarpGroups * WARP_SIZE;
  const int nSlots = nRanks > NSendSlots ? nRanks : NSendSlots;
  return configAlign<size_t>(static_cast<size_t>(nSlots) * sizeof(int), 128);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSendTmaBytes(int nTopk) {
  return DispatchMaxNWarpGroups * (dispatchPayloadStride(Hidden, nTopk) + sizeof(uint64_t));
}

template <int Hidden, int MaxWorkers>
MSCCLPP_HOST_DEVICE_INLINE constexpr int tmaWorkerCount() {
  static_assert(Hidden % 128 == 0);
  constexpr size_t workerBytes = static_cast<size_t>(Hidden) * sizeof(__bfloat16) + sizeof(uint64_t);
  constexpr int nWorkers = static_cast<int>((OptimizedDynamicSharedMemoryBytes - TmaWorkerControlBytes) / workerBytes);
  return nWorkers < MaxWorkers ? nWorkers : MaxWorkers;
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchRecvTmaBytes() {
  constexpr int NWorkers = tmaWorkerCount<Hidden, DispatchMaxNRecvTmaWorkers>();
  constexpr size_t tileBytes = static_cast<size_t>(Hidden) * sizeof(__bfloat16);
  return static_cast<size_t>(NWorkers) * (tileBytes + sizeof(uint64_t));
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedBytes(int nRanks, int nExperts, int nTopk) {
  const size_t controlBytes = dispatchSharedControlBytes(nRanks);
  const size_t sendBytes = dispatchSendTmaBytes<Hidden>(nTopk);
  const size_t recvBytes = dispatchRecvTmaBytes<Hidden>();
  const size_t tmaBytes = controlBytes + (sendBytes > recvBytes ? sendBytes : recvBytes);
  const size_t metadataBytes = static_cast<size_t>(nRanks + nExperts) * sizeof(int);
  return tmaBytes > metadataBytes ? tmaBytes : metadataBytes;
}

}  // namespace low_latency_opt
}  // namespace ep
}  // namespace mscclpp
