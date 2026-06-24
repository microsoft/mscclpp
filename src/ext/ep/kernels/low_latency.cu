// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Low-latency dispatch/combine kernels ported from DeepEP
// `csrc/kernels/internode_ll.cu` (branch `chhwang/dev-atomic-add-cleanup`).
//
// NVSHMEM/IBGDA device calls are replaced with MSCCL++ PortChannel device
// operations:
//
//   nvshmemx_barrier_all_block()              -> port-channel signal/wait ring
//   nvshmemi_ibgda_put_nbi_warp(...)          -> port_channel.put(...) (lane 0)
//   nvshmemi_ibgda_amo_nonfetch_add(...)      -> port_channel.atomicAdd(...)
//
// Addressing convention:
//   - `rdmaBufferPtr` is the base of the locally-registered RDMA buffer.
//   - Remote counter/buffer pointers written by the kernel are virtual
//     addresses that alias the corresponding offset inside each peer's
//     symmetric RDMA buffer. MSCCL++ needs those as offsets; we derive them
//     via `ptr - rdmaBufferPtr`.
//   - Port-channel layout built by `Buffer::sync()` in low-latency mode is
//     `handles[qp * num_peers + peer_idx]` where `peer_idx` is the dst rank's
//     position in the connected-peer map. In the recommended 1-GPU-per-node
//     LL topology, `peer_idx == dstRank`; see src/ext/ep/README.md.
//
// Validated on 2 nodes x 8 H100 GPUs via
// `test/python/ext/ep/test_low_latency_multirank.py`. Performance does NOT
// match IBGDA (host-proxy adds latency); see README for measurements.

#include <cooperative_groups.h>

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "api.cuh"
#include "common.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "quantization.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

namespace mscclpp {
namespace ep {

namespace low_latency {

// ---------------------------------------------------------------------------
// cleanLowLatencyBuffer
// ---------------------------------------------------------------------------

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__
    void cleanLowLatencyBuffer(int64_t* clean0, int numCleanInt0, int64_t* clean1, int numCleanInt1,
                               mscclpp::PortChannelDeviceHandle* portChannelHandles,
                               mscclpp::BaseMemoryChannelDeviceHandle* memoryChannelHandles, int rank, int numRanks,
                               int ranksPerIpcDomain) {
  // Barrier before cleaning (in case of unfinished chunked EP)
  channelBarrierBlock(portChannelHandles, memoryChannelHandles, rank, numRanks, ranksPerIpcDomain);

  // Clean
  auto threadId = static_cast<int>(threadIdx.x);
  for (int i = threadId; i < numCleanInt0; i += kNumThreads) clean0[i] = 0;
  for (int i = threadId; i < numCleanInt1; i += kNumThreads) clean1[i] = 0;

  // Barrier after cleaning (make sure low-latency mode work fine)
  channelBarrierBlock(portChannelHandles, memoryChannelHandles, rank, numRanks, ranksPerIpcDomain);
}

void cleanBuffers(int64_t* cleanup0, int cleanupSize0, int64_t* cleanup1, int cleanupSize1,
                  const TransportContext& transport, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;  // max EP shards
  const int kNumBlocks = 1;

  auto clean0 = cleanup0;
  auto numCleanInt0 = cleanupSize0;
  auto clean1 = cleanup1;
  auto numCleanInt1 = cleanupSize1;
  auto rank = transport.rank_;
  auto numRanks = transport.numRanks_;
  auto ranksPerIpcDomain = transport.ranksPerIpcDomain_;
  auto portChannelHandles = transport.portChannels_;
  auto memoryChannelHandles = transport.memoryChannels_;

  cleanLowLatencyBuffer<kThreadsPerBlock><<<dim3(kNumBlocks), dim3(kThreadsPerBlock), 0, stream>>>(
      clean0, numCleanInt0, clean1, numCleanInt1, portChannelHandles, memoryChannelHandles, rank, numRanks,
      ranksPerIpcDomain);
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

template <DType kDType>
struct DispatchDTypeTraits {};

template <>
struct DispatchDTypeTraits<DType::BF16> {
  using Type = nv_bfloat16;
};

template <>
struct DispatchDTypeTraits<DType::F8E4M3> {
  using Type = __nv_fp8_storage_t;
};

template <DType kInputDType, DType kOutputDType>
struct DispatchOutputVec {
  using SourceType = typename DispatchDTypeTraits<kInputDType>::Type;
  using Type =
      typename std::conditional<kInputDType == kOutputDType, int4, typename Fp8VectorType<SourceType>::Type>::type;
};

template <DType kInputDType, DType kOutputDType>
constexpr bool kDispatchNeedsScales = kInputDType != kOutputDType;

template <DType kInputDType, DType kOutputDType, int kNumPerChannels>
MSCCLPP_DEVICE_INLINE typename DispatchOutputVec<kInputDType, kOutputDType>::Type dispatchConvert(
    const int4& inputValue, float* scaleOut, int laneId) {
  using SourceType = typename DispatchDTypeTraits<kInputDType>::Type;
  using OutputVec = typename DispatchOutputVec<kInputDType, kOutputDType>::Type;

  if constexpr (kInputDType == kOutputDType) {
    return inputValue;
  } else {
    EP_STATIC_ASSERT(kInputDType == DType::BF16 && kOutputDType == DType::F8E4M3,
                     "Unsupported low-latency dispatch dtype conversion");
    return static_cast<OutputVec>(quantizeToFp8<SourceType, kNumPerChannels, __NV_E4M3>(inputValue, scaleOut, laneId));
  }
}

template <DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
MSCCLPP_DEVICE_INLINE void dispatchRecv(void* packedRecvX, float* packedRecvXScales, int* packedRecvSrcInfo,
                                        int64_t* packedRecvLayoutRange, int* packedRecvCount, void* rdmaRecvX,
                                        int64_t* rdmaRecvCount, int numMaxDispatchTokensPerRank, int numExperts,
                                        int numRanks, int numLocalExperts, int rank, size_t numBytesPerMsg,
                                        size_t hiddenInt4, size_t hiddenBytes, int numScales, int warpGroupId,
                                        int subWarpId, int laneId, int responsibleExpertIdx) {
  if (responsibleExpertIdx >= numExperts) return;

  const auto srcRank = responsibleExpertIdx / numLocalExperts;
  const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
  const auto rdmaRecvXUint8 = reinterpret_cast<uint8_t*>(rdmaRecvX) +
                              localExpertIdx * numRanks * numMaxDispatchTokensPerRank * numBytesPerMsg +
                              srcRank * numMaxDispatchTokensPerRank * numBytesPerMsg;
  const auto recvXInt4 =
      reinterpret_cast<int4*>(packedRecvX) + localExpertIdx * numRanks * numMaxDispatchTokensPerRank * hiddenInt4;
  const auto recvSrcInfo = packedRecvSrcInfo + localExpertIdx * numRanks * numMaxDispatchTokensPerRank;
  const auto recvRange = packedRecvLayoutRange + localExpertIdx * numRanks;

  __shared__ int sharedNumRecvTokens[kNumWarpGroups];
  __shared__ int sharedRecvTokenBeginIdx[kNumWarpGroups];

  int numRecvTokens;
  int recvTokenBeginIdx;
  EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
  if (subWarpId == 1 and laneId == 0) {
    const auto raw = waitSignalNonZero(rdmaRecvCount + localExpertIdx * numRanks + srcRank);
    numRecvTokens = static_cast<int>(-raw - 1);
    recvTokenBeginIdx = atomicAdd(packedRecvCount + localExpertIdx, numRecvTokens);
    sharedNumRecvTokens[warpGroupId] = numRecvTokens;
    sharedRecvTokenBeginIdx[warpGroupId] = recvTokenBeginIdx;
    recvRange[srcRank] = pack2<int, int64_t>(numRecvTokens, recvTokenBeginIdx);
  }
  asm volatile("bar.sync %0, %1;" ::"r"(warpGroupId + 2), "r"(kNumWarpsPerGroup * 32));
  numRecvTokens = sharedNumRecvTokens[warpGroupId];
  recvTokenBeginIdx = sharedRecvTokenBeginIdx[warpGroupId];

  EP_DEVICE_ASSERT(numScales <= 64);
  for (int i = subWarpId; i < numRecvTokens; i += kNumWarpsPerGroup) {
    const auto srcSrcIdx = reinterpret_cast<int*>(rdmaRecvXUint8 + i * numBytesPerMsg);
    if (laneId == 0) recvSrcInfo[recvTokenBeginIdx + i] = ld_nc_global(srcSrcIdx);
    __syncwarp();

    const auto srcData = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(srcSrcIdx) + sizeof(int4));
    const auto dstData = recvXInt4 + (recvTokenBeginIdx + i) * hiddenInt4;
    UNROLLED_WARP_COPY(7, laneId, hiddenInt4, dstData, srcData, ld_nc_global, st_na_global);

    if constexpr (kDispatchNeedsScales<DType::BF16, kOutputDType>) {
      const auto srcScales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(srcData) + hiddenBytes);
      const auto recvXScales = packedRecvXScales + localExpertIdx * numRanks * numMaxDispatchTokensPerRank * numScales;
      const auto dstScales = reinterpret_cast<float*>(recvXScales + recvTokenBeginIdx + i);
      const auto scaleStride = numRanks * numMaxDispatchTokensPerRank;
      auto scale0 = laneId < numScales ? ld_nc_global(srcScales + laneId) : 0;
      auto scale1 = (laneId + 32) < numScales ? ld_nc_global(srcScales + laneId + 32) : 0;
      laneId < numScales ? dstScales[laneId * scaleStride] = scale0 : 0.0f;
      (laneId + 32) < numScales ? dstScales[(laneId + 32) * scaleStride] = scale1 : 0.0f;
    }
  }
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
MSCCLPP_DEVICE_INLINE void dispatchSend(int* sharedNumTokensSentPerExpert, int* packedRecvCount, void* rdmaRecvX,
                                        int64_t* rdmaRecvCount, void* rdmaX, const void* x, const int64_t* topkIdx,
                                        int* atomicCounterPerExpert, int* atomicFinishCounterPerExpert,
                                        int64_t* nextClean, int numNextCleanInt, int numTokens,
                                        int numMaxDispatchTokensPerRank, int numTopk, int hidden, int numExperts,
                                        int rank, int numRanks, void* rdmaBufferPtr,
                                        mscclpp::PortChannelDeviceHandle* portChannelHandles,
                                        void* const* peerRdmaBases, int ranksPerIpcDomain) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto warpId = threadId / 32;
  const auto laneId = get_lane_id();
  const auto numSms = static_cast<int>(gridDim.x);
  const auto numWarps = kNumWarpGroups * kNumWarpsPerGroup;
  const auto numLocalExperts = numExperts / numRanks;
  const auto warpGroupId = warpId / kNumWarpsPerGroup;
  const auto subWarpId = warpId % kNumWarpsPerGroup;
  const auto responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  using SourceType = typename DispatchDTypeTraits<kInputDType>::Type;
  using OutputType = typename DispatchDTypeTraits<kOutputDType>::Type;
  constexpr int kNumPerChannels = 128;
  const int numScales = hidden / kNumPerChannels;
  const size_t hiddenBytes = hidden * sizeof(OutputType);
  using VecType = typename DispatchOutputVec<kInputDType, kOutputDType>::Type;
  const size_t numBytesPerMsg =
      sizeof(int4) + hiddenBytes + (kDispatchNeedsScales<kInputDType, kOutputDType> ? numScales * sizeof(float) : 0);
  const size_t numInt4PerMsg = numBytesPerMsg / sizeof(int4);

  if (warpId < numWarps - 1) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(SourceType);
    EP_DEVICE_ASSERT(hidden % kNumElemsPerRead == 0);
    EP_STATIC_ASSERT(kNumPerChannels % kNumElemsPerRead == 0, "Invalid vectorization");
    const auto sendWarps = (numWarps - 1) / numTopk * numTopk;
    const auto tokensPerRound = sendWarps / numTopk;
    const auto numThreads = (numWarps - 1) * 32;
    const size_t hiddenSourceInt4 = hidden / kNumElemsPerRead;

    for (int tokenBase = smId; tokenBase < numTokens; tokenBase += numSms * tokensPerRound) {
      for (int tokenGroupId = 0; tokenGroupId < tokensPerRound; ++tokenGroupId) {
        const auto tokenIdx = tokenBase + tokenGroupId * numSms;
        if (tokenIdx >= numTokens) continue;
        const auto xInt4 = reinterpret_cast<const int4*>(x) + tokenIdx * hiddenSourceInt4;
        const auto rdmaXSrcIdx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdmaX) + tokenIdx * numBytesPerMsg);
        const auto rdmaXVec = reinterpret_cast<VecType*>(reinterpret_cast<uint8_t*>(rdmaXSrcIdx) + sizeof(int4));
        const auto rdmaXScales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdmaXVec) + hiddenBytes);

        threadId == 0 ? (*rdmaXSrcIdx = tokenIdx) : 0;

        // Data conversion (BF16 or FP8). Keep the original full-payload-warp
        // pack path; only the following send stage is remapped.
        for (int i = threadId; i < hiddenSourceInt4; i += numThreads) {
          auto int4Value = __ldg(xInt4 + i);

          rdmaXVec[i] = dispatchConvert<kInputDType, kOutputDType, kNumPerChannels>(
              int4Value, &rdmaXScales[i * kNumElemsPerRead / kNumPerChannels], laneId);
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(numThreads));

      // Issue sends
      if (warpId < sendWarps) {
        const auto topkId = warpId % numTopk;
        const auto tokenGroupId = warpId / numTopk;
        const auto tokenIdx = tokenBase + tokenGroupId * numSms;
        const auto dstExpertIdx =
            tokenIdx < numTokens ? static_cast<int>(__ldg(topkIdx + tokenIdx * numTopk + topkId)) : -1;
        if (dstExpertIdx < 0) continue;
        const auto rdmaXSrcIdx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdmaX) + tokenIdx * numBytesPerMsg);
        int slotIdx = laneId == 0 ? atomicAdd(atomicCounterPerExpert + dstExpertIdx, 1) : 0;
        slotIdx = __shfl_sync(0xffffffff, slotIdx, 0);
        const auto dstRank = dstExpertIdx / numLocalExperts;
        const auto dstExpertLocalIdx = dstExpertIdx % numLocalExperts;
        const auto srcPtr = reinterpret_cast<uint64_t>(rdmaXSrcIdx);
        const auto dstPtr = reinterpret_cast<uint64_t>(rdmaRecvX) +
                            dstExpertLocalIdx * numRanks * numMaxDispatchTokensPerRank * numBytesPerMsg +
                            rank * numMaxDispatchTokensPerRank * numBytesPerMsg + slotIdx * numBytesPerMsg;
        if (dstRank != rank) {
          if (peerRdmaBases != nullptr && isIpcPeer(rank, dstRank, ranksPerIpcDomain)) {
            // Peer-mapped warp copy over NVLink (CUDA IPC).
            const auto peerDst = peerMappedPtrOf(dstPtr, peerRdmaBases, rdmaBufferPtr, dstRank);
            const auto* srcInt4Ptr = reinterpret_cast<const int4*>(srcPtr);
            const auto* dstInt4Ptr = reinterpret_cast<int4*>(peerDst);
            UNROLLED_WARP_COPY(8, laneId, numInt4PerMsg, dstInt4Ptr, srcInt4Ptr, ld_nc_global, st_na_global);
          } else {
            // MSCCL++ port-channel PUT (lane 0 issues one request).
            if (laneId == 0) {
              const auto dstOff = portChannelOffsetOf(dstPtr, rdmaBufferPtr);
              const auto srcOff = portChannelOffsetOf(srcPtr, rdmaBufferPtr);
              portChannelHandles[dstExpertLocalIdx * numRanks + dstRank].put(dstOff, srcOff, numBytesPerMsg);
            }
            __syncwarp();
          }
        } else {
          const auto* srcInt4Ptr = reinterpret_cast<const int4*>(srcPtr);
          const auto* dstInt4Ptr = reinterpret_cast<int4*>(dstPtr);
          UNROLLED_WARP_COPY(8, laneId, numInt4PerMsg, dstInt4Ptr, srcInt4Ptr, ld_nc_global, st_na_global);
        }

        __syncwarp();
        if (laneId == 0) atomicAddReleaseDevice(atomicFinishCounterPerExpert + dstExpertIdx, 1);
      }
    }
  } else if (warpId == numWarps - 1) {
    // The final warp does not send token payloads. It computes per-expert token
    // counts, initializes completion counters, and clears the next buffer's
    // signaling slots.
    EP_DEVICE_ASSERT(numSms > 1);
    if (smId == 0) {
      for (int i = laneId; i < numNextCleanInt; i += 32) nextClean[i] = 0;

      __syncwarp();
      for (int i = laneId; i < numExperts; i += 32)
        atomicAddReleaseDevice(atomicFinishCounterPerExpert + i, FINISHED_SUM_TAG);
    }

    int expertCount[kNumWarpGroups] = {0};
    const auto expertBeginIdx = smId * kNumWarpGroups;
    const auto expertEndIdx = min(expertBeginIdx + kNumWarpGroups, numExperts);

    for (int i = laneId; i < numTokens * numTopk; i += 32) {
      auto idx = static_cast<int>(__ldg(topkIdx + i));
      if (idx >= expertBeginIdx and idx < expertEndIdx) expertCount[idx - expertBeginIdx]++;
    }

#pragma unroll
    for (int i = expertBeginIdx; i < expertEndIdx; ++i) {
      auto sum = warp_reduce_sum(expertCount[i - expertBeginIdx]);
      if (laneId == 0) {
        sharedNumTokensSentPerExpert[i - expertBeginIdx] = sum;
        atomicAddReleaseDevice(atomicFinishCounterPerExpert + i, FINISHED_SUM_TAG - sum);
      }
    }
  }
  __syncthreads();

  // Issue count sends
  if (responsibleExpertIdx < numExperts and subWarpId == 0 and laneId == 0) {
    const auto dstRank = responsibleExpertIdx / numLocalExperts;
    const auto dstExpertLocalIdx = responsibleExpertIdx % numLocalExperts;
    const auto numTokensSent = sharedNumTokensSentPerExpert[responsibleExpertIdx - smId * kNumWarpGroups];

    while (ld_acquire_global(atomicFinishCounterPerExpert + responsibleExpertIdx) != FINISHED_SUM_TAG * 2);
    auto* counterPtr = rdmaRecvCount + dstExpertLocalIdx * numRanks + rank;
    auto* portChannelHandle = dstRank == rank ? nullptr : portChannelHandles + dstExpertLocalIdx * numRanks + dstRank;
    publishSingleWriterSignal(counterPtr, static_cast<int64_t>(-numTokensSent - 1), rank, dstRank, rdmaBufferPtr,
                              portChannelHandle, peerRdmaBases, ranksPerIpcDomain);

    atomicCounterPerExpert[responsibleExpertIdx] = 0;
    atomicFinishCounterPerExpert[responsibleExpertIdx] = 0;

    if (dstRank == 0) packedRecvCount[dstExpertLocalIdx] = 0;
  }
  __syncwarp();
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
__global__ __launch_bounds__(kNumWarpGroups* kNumWarpsPerGroup * 32, 1) void dispatch(
    void* packedRecvX, float* packedRecvXScales, int* packedRecvSrcInfo, int64_t* packedRecvLayoutRange,
    int* packedRecvCount, void* rdmaRecvX, int64_t* rdmaRecvCount, void* rdmaX, const void* x, const int64_t* topkIdx,
    int* atomicCounterPerExpert, int* atomicFinishCounterPerExpert, int64_t* nextClean, int numNextCleanInt,
    int numTokens, int numMaxDispatchTokensPerRank, int numTopk, int hidden, int numExperts, int rank, int numRanks,
    int phases, void* rdmaBufferPtr, mscclpp::PortChannelDeviceHandle* portChannelHandles, void* const* peerRdmaBases,
    int ranksPerIpcDomain) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto warpId = threadId / 32, laneId = get_lane_id();
  const auto numLocalExperts = numExperts / numRanks;
  const auto warpGroupId = warpId / kNumWarpsPerGroup;
  const auto subWarpId = warpId % kNumWarpsPerGroup;
  const auto responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  using OutputType = typename DispatchDTypeTraits<kOutputDType>::Type;
  constexpr int kNumPerChannels = 128;
  const int numScales = hidden / kNumPerChannels;
  const size_t hiddenBytes = hidden * sizeof(OutputType);
  const size_t hiddenInt4 = hiddenBytes / sizeof(int4);

  // Message package: hidden data, optional quantization scales, index at source
  const size_t numBytesPerMsg =
      sizeof(int4) + hiddenBytes + (kDispatchNeedsScales<kInputDType, kOutputDType> ? numScales * sizeof(float) : 0);
  EP_DEVICE_ASSERT(numBytesPerMsg % sizeof(int4) == 0);

  if (phases & LOW_LATENCY_SEND_PHASE) {
    __shared__ int sharedNumTokensSentPerExpert[kNumWarpGroups];
    dispatchSend<kInputDType, kOutputDType, kNumWarpGroups, kNumWarpsPerGroup>(
        sharedNumTokensSentPerExpert, packedRecvCount, rdmaRecvX, rdmaRecvCount, rdmaX, x, topkIdx,
        atomicCounterPerExpert, atomicFinishCounterPerExpert, nextClean, numNextCleanInt, numTokens,
        numMaxDispatchTokensPerRank, numTopk, hidden, numExperts, rank, numRanks, rdmaBufferPtr, portChannelHandles,
        peerRdmaBases, ranksPerIpcDomain);
  }

  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();

  dispatchRecv<kOutputDType, kNumWarpGroups, kNumWarpsPerGroup>(
      packedRecvX, packedRecvXScales, packedRecvSrcInfo, packedRecvLayoutRange, packedRecvCount, rdmaRecvX,
      rdmaRecvCount, numMaxDispatchTokensPerRank, numExperts, numRanks, numLocalExperts, rank, numBytesPerMsg,
      hiddenInt4, hiddenBytes, numScales, warpGroupId, subWarpId, laneId, responsibleExpertIdx);
}

constexpr int kDispatchNumMaxTopK = 9;
constexpr int kDispatchNumWarpGroups = 3;
constexpr int kDispatchNumWarpsPerGroup = 10;

struct DispatchLaunchArgs {
  cudaLaunchConfig_t* launchConfig;
  void* packedRecvX;
  float* packedRecvXScales;
  int* packedRecvSrcInfo;
  int64_t* packedRecvLayoutRange;
  int* packedRecvCount;
  void* rdmaRecvX;
  int64_t* rdmaRecvCount;
  void* rdmaX;
  const void* x;
  const int64_t* topkIdx;
  int* atomicCounterPerExpert;
  int* atomicFinishCounterPerExpert;
  int64_t* nextClean;
  int numNextCleanInt;
  int numTokens;
  int numMaxDispatchTokensPerRank;
  int numTopk;
  int hidden;
  int numExperts;
  int rank;
  int numRanks;
  int phases;
  void* rdmaBufferPtr;
  mscclpp::PortChannelDeviceHandle* portChannelHandles;
  void* const* peerRdmaBases;
  int ranksPerIpcDomain;
};

template <DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
void launchDispatchKernel(const DispatchLaunchArgs& args) {
  auto dispatchFunc = dispatch<DType::BF16, kOutputDType, kNumWarpGroups, kNumWarpsPerGroup>;
  CUDA_CHECK(cudaLaunchKernelEx(args.launchConfig, dispatchFunc, args.packedRecvX, args.packedRecvXScales,
                                args.packedRecvSrcInfo, args.packedRecvLayoutRange, args.packedRecvCount,
                                args.rdmaRecvX, args.rdmaRecvCount, args.rdmaX, args.x, args.topkIdx,
                                args.atomicCounterPerExpert, args.atomicFinishCounterPerExpert, args.nextClean,
                                args.numNextCleanInt, args.numTokens, args.numMaxDispatchTokensPerRank, args.numTopk,
                                args.hidden, args.numExperts, args.rank, args.numRanks, args.phases, args.rdmaBufferPtr,
                                args.portChannelHandles, args.peerRdmaBases, args.ranksPerIpcDomain));
}

template <DType kOutputDType>
void launchDispatchForOutput(const DispatchLaunchArgs& args) {
  launchDispatchKernel<kOutputDType, kDispatchNumWarpGroups, kDispatchNumWarpsPerGroup>(args);
}

void launchDispatchForOutputDType(const DispatchLaunchArgs& args, DType outputDType) {
  switch (outputDType) {
    case DType::BF16:
      launchDispatchForOutput<DType::BF16>(args);
      break;
    case DType::F8E4M3:
      launchDispatchForOutput<DType::F8E4M3>(args);
      break;
    default:
      EP_HOST_ASSERT(false && "Unsupported low-latency dispatch output dtype");
  }
}

void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
              const void* input, const int64_t* topkIdx, const DispatchConfig& config, const BufferSet& currentBuffer,
              const BufferSet& nextBuffer, const TransportContext& transport, void* workspace, cudaStream_t stream,
              Phase phase) {
  // Unpack configuration
  auto packedRecvX = output;
  auto packedRecvXScales = outputScales;
  auto packedRecvSrcInfo = outputSrcInfo;
  auto packedRecvLayoutRange = outputLayout;
  auto packedRecvCount = outputCount;
  auto rdmaRecvX = currentBuffer.recvDataBuffer_;
  auto rdmaRecvCount = currentBuffer.recvCountBuffer_;
  auto rdmaX = currentBuffer.sendDataBuffer_;
  auto x = input;
  auto nextClean = nextBuffer.cleanupRegion_;
  auto numNextCleanInt = nextBuffer.cleanupSize_;
  auto numTokens = config.numTokens_;
  auto hidden = config.hidden_;
  auto numMaxDispatchTokensPerRank = config.numMaxTokensPerRank_;
  auto numTopk = config.numTopk_;
  auto numExperts = config.numExperts_;
  auto rank = transport.rank_;
  auto numRanks = transport.numRanks_;
  auto inputDType = config.inputDType_;
  auto outputDType = config.outputDType_;
  auto phases = static_cast<int>(phase);
  auto rdmaBufferPtr = transport.rdmaBufferBase_;
  auto portChannelHandles = transport.portChannels_;
  auto peerRdmaBases = transport.peerBases_;
  auto ranksPerIpcDomain = transport.ranksPerIpcDomain_;
  EP_STATIC_ASSERT(kDispatchNumMaxTopK + 1 <= kDispatchNumWarpGroups * kDispatchNumWarpsPerGroup,
                   "Too many top-k selections");

  const auto numWarps = kDispatchNumWarpGroups * kDispatchNumWarpsPerGroup;
  const auto numSms = cell_div(numExperts, kDispatchNumWarpGroups);
  EP_HOST_ASSERT(numTopk > 0);
  EP_HOST_ASSERT(numTopk <= kDispatchNumMaxTopK);

  auto atomicCounterPerExpert = reinterpret_cast<int*>(workspace);
  auto atomicFinishCounterPerExpert = atomicCounterPerExpert + numExperts;
  EP_HOST_ASSERT(numExperts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(inputDType == DType::BF16);
  EP_HOST_ASSERT(hidden % 128 == 0);

  cudaLaunchConfig_t cfg = {dim3(numSms), dim3(numWarps * 32), 0, stream, nullptr, 0};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  DispatchLaunchArgs args{.launchConfig = &cfg,
                          .packedRecvX = packedRecvX,
                          .packedRecvXScales = packedRecvXScales,
                          .packedRecvSrcInfo = packedRecvSrcInfo,
                          .packedRecvLayoutRange = packedRecvLayoutRange,
                          .packedRecvCount = packedRecvCount,
                          .rdmaRecvX = rdmaRecvX,
                          .rdmaRecvCount = rdmaRecvCount,
                          .rdmaX = rdmaX,
                          .x = x,
                          .topkIdx = topkIdx,
                          .atomicCounterPerExpert = atomicCounterPerExpert,
                          .atomicFinishCounterPerExpert = atomicFinishCounterPerExpert,
                          .nextClean = nextClean,
                          .numNextCleanInt = numNextCleanInt,
                          .numTokens = numTokens,
                          .numMaxDispatchTokensPerRank = numMaxDispatchTokensPerRank,
                          .numTopk = numTopk,
                          .hidden = hidden,
                          .numExperts = numExperts,
                          .rank = rank,
                          .numRanks = numRanks,
                          .phases = phases,
                          .rdmaBufferPtr = rdmaBufferPtr,
                          .portChannelHandles = portChannelHandles,
                          .peerRdmaBases = peerRdmaBases,
                          .ranksPerIpcDomain = ranksPerIpcDomain};

  launchDispatchForOutputDType(args, outputDType);
}

// ---------------------------------------------------------------------------
// combine
// ---------------------------------------------------------------------------

template <DType kInputDType, int kHidden>
MSCCLPP_DEVICE_INLINE void copyCombineInputToBf16(int4* dst, const uint8_t* src, const float* scales, int scaleStride,
                                                  int laneId) {
  constexpr int kNumBf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kHiddenBf16Int4 = kHidden / kNumBf16PerInt4;

  if constexpr (kInputDType == DType::BF16) {
    const auto srcInt4 = reinterpret_cast<const int4*>(src);
    UNROLLED_WARP_COPY(7, laneId, kHiddenBf16Int4, dst, srcInt4, ld_nc_global, st_na_global);
  } else {
    EP_STATIC_ASSERT(kInputDType == DType::F8E4M3, "Unsupported low-latency combine input dtype");
    const auto srcFp8 = reinterpret_cast<const __nv_fp8_storage_t*>(src);
    EP_DEVICE_ASSERT(scales != nullptr);

    for (int i = laneId; i < kHiddenBf16Int4; i += 32) {
      int4 bf16Pack;
      auto bf16Values = reinterpret_cast<nv_bfloat16*>(&bf16Pack);
#pragma unroll
      for (int j = 0; j < kNumBf16PerInt4; ++j) {
        const int elemIdx = i * kNumBf16PerInt4 + j;
        const int scaleIdx = elemIdx / 128;
        bf16Values[j] =
            static_cast<nv_bfloat16>(dequantizeFp8<__NV_E4M3>(srcFp8[elemIdx], scales[scaleIdx * scaleStride]));
      }
      st_na_global(dst + i, bf16Pack);
    }
  }
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden,
          int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups* kNumWarpsPerGroup * 32, 1) void combine(
    void* combinedX, void* rdmaRecvX, int64_t* rdmaRecvFlag, void* rdmaSendX, const void* x, const float* xScales,
    const int64_t* topkIdx, const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
    int64_t* nextClean, int numNextCleanInt, int* atomicCleanFlag, int numCombinedTokens, int hidden, int numTopk,
    int numMaxDispatchTokensPerRank, int numExperts, int rank, int numRanks, int phases, bool zeroCopy,
    void* rdmaBufferPtr, mscclpp::PortChannelDeviceHandle* portChannelHandles, void* const* peerRdmaBases,
    int ranksPerIpcDomain) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto numSms = static_cast<int>(gridDim.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto numThreads = static_cast<int>(blockDim.x);
  const auto warpId = threadId / 32, laneId = get_lane_id();
  const auto numLocalExperts = numExperts / numRanks;
  const auto warpGroupId = warpId / kNumWarpsPerGroup;
  const auto subWarpId = warpId % kNumWarpsPerGroup;
  const auto responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  EP_STATIC_ASSERT(kOutputDType == DType::BF16, "Only BF16 low-latency combine output is supported");
  using InputType = typename DispatchDTypeTraits<kInputDType>::Type;
  using OutputType = typename DispatchDTypeTraits<kOutputDType>::Type;

  constexpr int kNumBf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  const size_t hiddenBf16Int4 = kHidden / kNumBf16PerInt4;

  constexpr size_t numInputBytes = kHidden * sizeof(InputType);
  constexpr size_t numBytesPerSlot = kHidden * sizeof(OutputType);
  EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

  if (phases & LOW_LATENCY_SEND_PHASE) {
    if (smId == 0 and warpGroupId == 0 and subWarpId == 0) {
      for (int i = laneId; i < numNextCleanInt; i += 32) nextClean[i] = 0;

      __syncwarp();
      if (laneId == 0) atomicAddReleaseDevice(atomicCleanFlag, numExperts);
    }

    if (responsibleExpertIdx < numExperts) {
      const auto dstRank = responsibleExpertIdx / numLocalExperts;
      const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
      const auto globalExpertIdx = rank * numLocalExperts + localExpertIdx;
      const auto layout = __ldg(layoutRange + localExpertIdx * numRanks + dstRank);
      const int scaleStride = numRanks * numMaxDispatchTokensPerRank;
      const int numScales = kHidden / 128;
      const auto localX = reinterpret_cast<const uint8_t*>(x) + localExpertIdx * scaleStride * numInputBytes;
      const auto localXScales = xScales == nullptr ? nullptr : xScales + localExpertIdx * numScales * scaleStride;
      const auto localSrcInfo = srcInfo + localExpertIdx * numRanks * numMaxDispatchTokensPerRank;
      const auto rdmaSendXVec = reinterpret_cast<uint8_t*>(rdmaSendX) +
                                localExpertIdx * numRanks * numMaxDispatchTokensPerRank * numBytesPerSlot;

      int offset, numTokensToSend;
      unpack2(layout, numTokensToSend, offset);

      for (int tokenIdx = offset + subWarpId; tokenIdx < offset + numTokensToSend; tokenIdx += kNumWarpsPerGroup) {
        const auto xRow = localX + tokenIdx * numInputBytes;
        const auto xScaleRow = localXScales == nullptr ? nullptr : localXScales + tokenIdx;
        const auto rdmaSendTypeRow = reinterpret_cast<int*>(rdmaSendXVec + tokenIdx * numBytesPerSlot);
        const auto rdmaSendXVecRow = reinterpret_cast<uint8_t*>(rdmaSendTypeRow);

        auto srcIdx = __ldg(localSrcInfo + tokenIdx);
        const auto bufPtr = reinterpret_cast<int64_t>(rdmaSendXVecRow);
        const auto dstPtr = reinterpret_cast<uint64_t>(rdmaRecvX) +
                            (globalExpertIdx * numMaxDispatchTokensPerRank + srcIdx) * numBytesPerSlot;
        if (dstRank == rank) {
          const auto dstInt4Ptr = reinterpret_cast<int4*>(dstPtr);
          copyCombineInputToBf16<kInputDType, kHidden>(dstInt4Ptr, xRow, xScaleRow, scaleStride, laneId);
        } else {
          if (peerRdmaBases != nullptr && isIpcPeer(rank, dstRank, ranksPerIpcDomain)) {
            // Peer-mapped warp copy over NVLink. `zeroCopy` is irrelevant
            // on this path because we skip the rdma_send staging buffer.
            const auto peerDst = peerMappedPtrOf(dstPtr, peerRdmaBases, rdmaBufferPtr, dstRank);
            const auto peerDstInt4 = reinterpret_cast<int4*>(peerDst);
            copyCombineInputToBf16<kInputDType, kHidden>(peerDstInt4, xRow, xScaleRow, scaleStride, laneId);
          } else {
            const auto bufInt4Ptr = reinterpret_cast<int4*>(bufPtr);
            if constexpr (kInputDType == DType::BF16) {
              if (not zeroCopy)
                copyCombineInputToBf16<kInputDType, kHidden>(bufInt4Ptr, xRow, xScaleRow, scaleStride, laneId);
            } else {
              copyCombineInputToBf16<kInputDType, kHidden>(bufInt4Ptr, xRow, xScaleRow, scaleStride, laneId);
            }
            // MSCCL++ port-channel PUT.
            if (laneId == 0) {
              const auto dstOff = portChannelOffsetOf(dstPtr, rdmaBufferPtr);
              const auto srcOff = portChannelOffsetOf(static_cast<uint64_t>(bufPtr), rdmaBufferPtr);
              portChannelHandles[localExpertIdx * numRanks + dstRank].put(dstOff, srcOff, hidden * sizeof(OutputType));
            }
            __syncwarp();
          }
        }
      }

      EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
      asm volatile("bar.sync %0, %1;" ::"r"(warpGroupId + 1), "r"(kNumWarpsPerGroup * 32));
      if (subWarpId == 1 and laneId == 0) {
        while (ld_acquire_global(atomicCleanFlag) == 0);
        auto* flagPtr = rdmaRecvFlag + globalExpertIdx;
        auto* portChannelHandle = dstRank == rank ? nullptr : portChannelHandles + localExpertIdx * numRanks + dstRank;
        publishSingleWriterSignal(flagPtr, static_cast<int64_t>(1), rank, dstRank, rdmaBufferPtr, portChannelHandle,
                                  peerRdmaBases, ranksPerIpcDomain);
        atomicAddReleaseDevice(atomicCleanFlag, -1);
      }
      __syncwarp();
    }
  }

  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  if (responsibleExpertIdx < numExperts) {
    EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
    if (subWarpId == 0 and laneId == 0) waitSignalNonZero(rdmaRecvFlag + responsibleExpertIdx);
  }
  cg::this_grid().sync();

  EP_DEVICE_ASSERT(numTopk <= 32 and hiddenBf16Int4 <= numThreads);
  EP_STATIC_ASSERT(kHidden % (32 * kNumBf16PerInt4) == 0, "Invalid vectorization");
  if (threadId < hiddenBf16Int4) {
    for (int tokenIdx = smId; tokenIdx < numCombinedTokens; tokenIdx += numSms) {
      int regTopkIdx[kNumMaxTopk];
      float regTopkWeights[kNumMaxTopk];
      for (int i = 0; i < numTopk; ++i) {
        regTopkIdx[i] = static_cast<int>(__ldg(topkIdx + tokenIdx * numTopk + i));
        regTopkWeights[i] = __ldg(topkWeights + tokenIdx * numTopk + i);
      }

      float combinedValues[kNumBf16PerInt4] = {0.0f};
      for (int i = 0; i < numTopk; ++i)
        if (regTopkIdx[i] >= 0) {
          auto rdmaBufferType =
              reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdmaRecvX) +
                                           (regTopkIdx[i] * numMaxDispatchTokensPerRank + tokenIdx) * numBytesPerSlot);
          auto rdmaBufferRow = reinterpret_cast<const uint8_t*>(rdmaBufferType);

          auto xVec = ld_nc_global(reinterpret_cast<const int4*>(rdmaBufferRow) + threadId);
          const auto xValues = reinterpret_cast<nv_bfloat16*>(&xVec);
#pragma unroll
          for (int j = 0; j < kNumBf16PerInt4; ++j)
            combinedValues[j] += static_cast<float>(xValues[j]) * regTopkWeights[i];
        }

      int4 combinedInt4;
      auto combinedOutput = reinterpret_cast<OutputType*>(&combinedInt4);
#pragma unroll
      for (int j = 0; j < kNumBf16PerInt4; ++j) combinedOutput[j] = static_cast<OutputType>(combinedValues[j]);
      (reinterpret_cast<int4*>(combinedX) + tokenIdx * hiddenBf16Int4)[threadId] = combinedInt4;
    }
  }
}

void combine(void* output, const void* input, const float* inputScales, const int64_t* topkIdx,
             const float* topkWeights, const int* srcInfo, const int64_t* layoutRange, const CombineConfig& config,
             const BufferSet& currentBuffer, const BufferSet& nextBuffer, const TransportContext& transport,
             void* workspace, cudaStream_t stream, Phase phase) {
  // Unpack configuration
  auto combinedX = output;
  auto rdmaRecvX = currentBuffer.recvDataBuffer_;
  auto rdmaRecvFlag = currentBuffer.recvCountBuffer_;
  auto rdmaSendX = currentBuffer.sendDataBuffer_;
  auto x = input;
  auto xScales = inputScales;
  auto nextClean = nextBuffer.cleanupRegion_;
  auto numNextCleanInt = nextBuffer.cleanupSize_;
  auto numCombinedTokens = config.numCombinedTokens_;
  auto hidden = config.hidden_;
  auto numMaxDispatchTokensPerRank = config.numMaxTokensPerRank_;
  auto numTopk = config.numTopk_;
  auto numExperts = config.numExperts_;
  auto rank = transport.rank_;
  auto numRanks = transport.numRanks_;
  auto inputDType = config.inputDType_;
  auto outputDType = config.outputDType_;
  auto zeroCopy = config.zeroCopy_;
  auto phases = static_cast<int>(phase);
  auto rdmaBufferPtr = transport.rdmaBufferBase_;
  auto portChannelHandles = transport.portChannels_;
  auto peerRdmaBases = transport.peerBases_;
  auto ranksPerIpcDomain = transport.ranksPerIpcDomain_;
  constexpr int kNumWarpGroups = 3;
  constexpr int kNumWarpsPerGroup = 10;
  constexpr int kNumMaxTopk = 9;

  const auto numWarps = kNumWarpGroups * kNumWarpsPerGroup;
  const auto numSmsBase = cell_div(numExperts, kNumWarpGroups);
  const auto numSms = numSmsBase;

  auto atomicCleanFlag = reinterpret_cast<int*>(workspace);
  EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(numTopk <= kNumMaxTopk);
  EP_HOST_ASSERT(outputDType == DType::BF16);

#define COMBINE_LAUNCH(input_dtype, hidden_case)                                                                       \
  {                                                                                                                    \
    auto combineFunc = combine<input_dtype, DType::BF16, kNumWarpGroups, kNumWarpsPerGroup, hidden_case, kNumMaxTopk>; \
    LAUNCH_KERNEL(&cfg, combineFunc, combinedX, rdmaRecvX, rdmaRecvFlag, rdmaSendX, x, xScales, topkIdx, topkWeights,  \
                  srcInfo, layoutRange, nextClean, numNextCleanInt, atomicCleanFlag, numCombinedTokens, hidden,        \
                  numTopk, numMaxDispatchTokensPerRank, numExperts, rank, numRanks, phases, zeroCopy, rdmaBufferPtr,   \
                  portChannelHandles, peerRdmaBases, ranksPerIpcDomain);                                               \
  }

#define COMBINE_LAUNCH_CASE(hidden_case)           \
  {                                                \
    if (inputDType == DType::BF16) {               \
      COMBINE_LAUNCH(DType::BF16, hidden_case);    \
    } else {                                       \
      EP_HOST_ASSERT(inputDType == DType::F8E4M3); \
      COMBINE_LAUNCH(DType::F8E4M3, hidden_case);  \
    }                                              \
  }                                                \
  break

  SETUP_LAUNCH_CONFIG(numSms, numWarps * 32, stream);
  SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
#undef COMBINE_LAUNCH
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
