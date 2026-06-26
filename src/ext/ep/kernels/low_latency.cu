// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Low-latency dispatch/combine kernels adapted from DeepEP
// `csrc/kernels/internode_ll.cu`.
//
// Transport mapping in this MSCCL++ implementation:
//   - Local rank: direct global-memory copy/store.
//   - CUDA-IPC peer: direct store through the imported peer RDMA-buffer VA.
//   - Non-IPC peer: MSCCL++ PortChannel PUT from the locally registered RDMA
//     staging buffer; lane 0 issues the proxy request.
//   - Count/flag signals: direct release store for local/IPC peers, otherwise
//     PortChannel 64-bit atomicAdd.
//
// Addressing convention:
//   - `rdmaBufferPtr` is the base of this rank's registered RDMA buffer.
//   - Pointers into a peer's symmetric RDMA buffer use the same offset as the
//     local pointer. PortChannel needs offsets, so the kernels derive them as
//     `ptr - rdmaBufferPtr`; CUDA-IPC paths add the same offset to
//     `peerRdmaBases[peerRank]`.
//   - `portChannelHandles[localExpertIdx * numRanks + peerRank]` is the
//     PortChannel used for that local expert / peer-rank transfer.

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
    void cleanLowLatencyBuffer(int64_t* buffer0, int numInt0, int64_t* buffer1, int numInt1,
                               mscclpp::PortChannelDeviceHandle* portChannelHandles,
                               mscclpp::BaseMemoryChannelDeviceHandle* memoryChannelHandles, int rank, int numRanks,
                               int ranksPerIpcDomain) {
  // Barrier before cleaning in case the previous low-latency epoch is still draining.
  channelBarrierBlock(portChannelHandles, memoryChannelHandles, rank, numRanks, ranksPerIpcDomain);

  // Clean
  auto threadId = static_cast<int>(threadIdx.x);
  for (int i = threadId; i < numInt0; i += kNumThreads) buffer0[i] = 0;
  for (int i = threadId; i < numInt1; i += kNumThreads) buffer1[i] = 0;

  // Barrier after cleaning so every rank observes zeroed signaling slots.
  channelBarrierBlock(portChannelHandles, memoryChannelHandles, rank, numRanks, ranksPerIpcDomain);
}

void cleanBuffers(int64_t* buffer0, int numInt0, int64_t* buffer1, int numInt1, const TransportContext& transport,
                  cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;  // max EP shards
  const int kNumBlocks = 1;

  auto rank = transport.rank_;
  auto numRanks = transport.numRanks_;
  auto ranksPerIpcDomain = transport.ranksPerIpcDomain_;
  auto portChannelHandles = transport.portChannels_;
  auto memoryChannelHandles = transport.memoryChannels_;
  cleanLowLatencyBuffer<kThreadsPerBlock><<<dim3(kNumBlocks), dim3(kThreadsPerBlock), 0, stream>>>(
      buffer0, numInt0, buffer1, numInt1, portChannelHandles, memoryChannelHandles, rank, numRanks, ranksPerIpcDomain);
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

// Keep input and output dtype separate. Current launch path only instantiates
// BF16 input, but this keeps room for pre-quantized input with explicit scales.
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
    static_assert(kInputDType == DType::BF16 && kOutputDType == DType::F8E4M3,
                  "Unsupported low-latency dispatch dtype conversion");
    return static_cast<OutputVec>(quantizeToFp8<SourceType, kNumPerChannels, __NV_E4M3>(inputValue, scaleOut, laneId));
  }
}

template <DType kInputDType, DType kOutputDType>
MSCCLPP_DEVICE_INLINE void copyScales(float* outputScales, const int4* stagedPayload, int localExpertIdx,
                                      int recvTokenIdx, int numRanks, int numMaxDispatchTokensPerRank,
                                      size_t hiddenBytes, int numScales, int laneId) {
  if constexpr (kDispatchNeedsScales<kInputDType, kOutputDType>) {
    static_assert(kInputDType == DType::BF16 && kOutputDType == DType::F8E4M3,
                  "Unsupported low-latency dispatch scale copy");

    EP_DEVICE_ASSERT(outputScales != nullptr);
    EP_DEVICE_ASSERT(numScales <= 64);

    const auto stagedScales =
        reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(stagedPayload) + hiddenBytes);
    const auto slotsPerExpert = numRanks * numMaxDispatchTokensPerRank;
    const auto outputScaleBase = outputScales + localExpertIdx * slotsPerExpert * numScales;
    const auto outputScaleRow = outputScaleBase + recvTokenIdx;

    auto scale0 = laneId < numScales ? ld_nc_global(stagedScales + laneId) : 0;
    auto scale1 = (laneId + WARP_SIZE) < numScales ? ld_nc_global(stagedScales + laneId + WARP_SIZE) : 0;
    laneId < numScales ? outputScaleRow[laneId * slotsPerExpert] = scale0 : 0.0f;
    (laneId + WARP_SIZE) < numScales ? outputScaleRow[(laneId + WARP_SIZE) * slotsPerExpert] = scale1 : 0.0f;
  }
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
MSCCLPP_DEVICE_INLINE void dispatchRecv(void* output, float* outputScales, int* outputSrcInfo,
                                        int64_t* outputLayoutRange, int* outputCount, void* stagedRecv,
                                        int64_t* stagedRecvCountBuffer, int numMaxDispatchTokensPerRank, int numExperts,
                                        int numRanks, int numLocalExperts, int rank, size_t numBytesPerMsg,
                                        size_t hiddenInt4, size_t hiddenBytes, int numScales,
                                        DispatchLayout dispatchLayout, int warpGroupId, int subWarpId, int laneId,
                                        int responsibleExpertIdx) {
  if (responsibleExpertIdx >= numExperts) return;

  const auto sourceRank = responsibleExpertIdx / numLocalExperts;
  const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
  (void)dispatchLayout;  // EXPERT_MAJOR and FLAT share the same physical order.
  const auto stagedMsgBase = reinterpret_cast<uint8_t*>(stagedRecv) +
                             localExpertIdx * numRanks * numMaxDispatchTokensPerRank * numBytesPerMsg +
                             sourceRank * numMaxDispatchTokensPerRank * numBytesPerMsg;
  const auto outputPayload =
      reinterpret_cast<int4*>(output) + localExpertIdx * numRanks * numMaxDispatchTokensPerRank * hiddenInt4;
  const auto packedSrcInfo = outputSrcInfo + localExpertIdx * numRanks * numMaxDispatchTokensPerRank;
  const auto outputLayout = outputLayoutRange + localExpertIdx * numRanks;

  __shared__ int sharedNumRecvTokens[kNumWarpGroups];
  __shared__ int sharedRecvTokenBeginIdx[kNumWarpGroups];

  int numRecvTokens;
  int recvTokenBeginIdx;
  static_assert(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
  if (subWarpId == 0 and laneId == 0) {
    const auto raw = waitSignalNonZero(stagedRecvCountBuffer + localExpertIdx * numRanks + sourceRank);
    numRecvTokens = static_cast<int>(-raw - 1);
    recvTokenBeginIdx = atomicAdd(outputCount + localExpertIdx, numRecvTokens);
    sharedNumRecvTokens[warpGroupId] = numRecvTokens;
    sharedRecvTokenBeginIdx[warpGroupId] = recvTokenBeginIdx;
    outputLayout[sourceRank] = pack2<int, int64_t>(numRecvTokens, recvTokenBeginIdx);
  }
  asm volatile("bar.sync %0, %1;" ::"r"(warpGroupId + 2), "r"(kNumWarpsPerGroup * WARP_SIZE));
  numRecvTokens = sharedNumRecvTokens[warpGroupId];
  recvTokenBeginIdx = sharedRecvTokenBeginIdx[warpGroupId];

  EP_DEVICE_ASSERT(numScales <= 64);
  for (int i = subWarpId; i < numRecvTokens; i += kNumWarpsPerGroup) {
    const auto stagedMsg = reinterpret_cast<int*>(stagedMsgBase + i * numBytesPerMsg);
    if (laneId == 0) packedSrcInfo[recvTokenBeginIdx + i] = ld_nc_global(stagedMsg);
    __syncwarp();

    const auto stagedPayload = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(stagedMsg) + sizeof(int4));
    const auto outputPayloadRow = outputPayload + (recvTokenBeginIdx + i) * hiddenInt4;
    UNROLLED_WARP_COPY(7, laneId, hiddenInt4, outputPayloadRow, stagedPayload, ld_nc_global, st_na_global);

    copyScales<kInputDType, kOutputDType>(outputScales, stagedPayload, localExpertIdx, recvTokenBeginIdx + i, numRanks,
                                          numMaxDispatchTokensPerRank, hiddenBytes, numScales, laneId);
  }
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
MSCCLPP_DEVICE_INLINE void dispatchSend(int* sharedNumTokensSentPerExpert, int* outputCount, void* stagedRecv,
                                        int64_t* stagedRecvCountBuffer, void* stagedSend, const void* input,
                                        const int64_t* topkIdx, int* atomicCounterPerExpert,
                                        int* atomicFinishCounterPerExpert, int64_t* nextClean, int numNextCleanInt,
                                        int numTokens, int numMaxDispatchTokensPerRank, int numTopk, int hidden,
                                        int numExperts, int rank, int numRanks, void* rdmaBufferPtr,
                                        mscclpp::PortChannelDeviceHandle* portChannelHandles,
                                        void* const* peerRdmaBases, int ranksPerIpcDomain) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto warpId = threadId / WARP_SIZE;
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
    static_assert(kNumPerChannels % kNumElemsPerRead == 0, "Invalid vectorization");
    const auto numPayloadWarps = numWarps - 1;
    const auto numSendWarps = numPayloadWarps / numTopk * numTopk;
    const auto numTokensPerSendRound = numSendWarps / numTopk;
    const auto numPayloadThreads = numPayloadWarps * WARP_SIZE;
    const size_t hiddenSourceInt4 = hidden / kNumElemsPerRead;

    // Keep the pack/cast path unchanged: all payload warps cooperatively stage
    // a small tile of tokens into stagedSend. Only the send stage below is remapped
    // so each token in the tile gets exactly numTopk sender warps.
    for (int tokenBase = smId; tokenBase < numTokens; tokenBase += numSms * numTokensPerSendRound) {
      for (int tokenGroupId = 0; tokenGroupId < numTokensPerSendRound; ++tokenGroupId) {
        const auto tokenIdx = tokenBase + tokenGroupId * numSms;
        if (tokenIdx >= numTokens) continue;
        const auto inputVec = reinterpret_cast<const int4*>(input) + tokenIdx * hiddenSourceInt4;
        const auto stagedSendMsg =
            reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(stagedSend) + tokenIdx * numBytesPerMsg);
        const auto stagedSendPayload =
            reinterpret_cast<VecType*>(reinterpret_cast<uint8_t*>(stagedSendMsg) + sizeof(int4));
        const auto stagedSendScales =
            reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(stagedSendPayload) + hiddenBytes);

        threadId == 0 ? (*stagedSendMsg = tokenIdx) : 0;

        for (int i = threadId; i < hiddenSourceInt4; i += numPayloadThreads) {
          auto int4Value = __ldg(inputVec + i);

          stagedSendPayload[i] = dispatchConvert<kInputDType, kOutputDType, kNumPerChannels>(
              int4Value, &stagedSendScales[i * kNumElemsPerRead / kNumPerChannels], laneId);
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(numPayloadThreads));

      // Send with a warp layout that is a multiple of top-k:
      //   warp 0..numTopk-1       -> token group 0, top-k 0..numTopk-1
      //   warp numTopk..2*numTopk -> token group 1, top-k 0..numTopk-1
      // Any leftover payload warps still participate in staging, but skip send.
      if (warpId < numSendWarps) {
        const auto topkId = warpId % numTopk;
        const auto tokenGroupId = warpId / numTopk;
        const auto tokenIdx = tokenBase + tokenGroupId * numSms;
        const auto dstExpertIdx =
            tokenIdx < numTokens ? static_cast<int>(__ldg(topkIdx + tokenIdx * numTopk + topkId)) : -1;
        if (dstExpertIdx < 0) continue;
        const auto stagedSendMsg =
            reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(stagedSend) + tokenIdx * numBytesPerMsg);
        int slotIdx = laneId == 0 ? atomicAdd(atomicCounterPerExpert + dstExpertIdx, 1) : 0;
        slotIdx = __shfl_sync(0xffffffff, slotIdx, 0);
        const auto dstRank = dstExpertIdx / numLocalExperts;
        const auto dstExpertLocalIdx = dstExpertIdx % numLocalExperts;
        const auto srcPtr = reinterpret_cast<uint64_t>(stagedSendMsg);
        const auto dstPtr = reinterpret_cast<uint64_t>(stagedRecv) +
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
      for (int i = laneId; i < numNextCleanInt; i += WARP_SIZE) nextClean[i] = 0;

      __syncwarp();
      for (int i = laneId; i < numExperts; i += WARP_SIZE)
        atomicAddReleaseDevice(atomicFinishCounterPerExpert + i, FINISHED_SUM_TAG);
    }

    int expertCount[kNumWarpGroups] = {0};
    const auto expertBeginIdx = smId * kNumWarpGroups;
    const auto expertEndIdx = min(expertBeginIdx + kNumWarpGroups, numExperts);

    for (int i = laneId; i < numTokens * numTopk; i += WARP_SIZE) {
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
    auto* counterPtr = stagedRecvCountBuffer + dstExpertLocalIdx * numRanks + rank;
    auto* portChannelHandle = dstRank == rank ? nullptr : portChannelHandles + dstExpertLocalIdx * numRanks + dstRank;
    publishSingleWriterSignal(counterPtr, static_cast<int64_t>(-numTokensSent - 1), rank, dstRank, rdmaBufferPtr,
                              portChannelHandle, peerRdmaBases, ranksPerIpcDomain);

    atomicCounterPerExpert[responsibleExpertIdx] = 0;
    atomicFinishCounterPerExpert[responsibleExpertIdx] = 0;

    if (dstRank == 0) outputCount[dstExpertLocalIdx] = 0;
  }
  __syncwarp();
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
__global__ __launch_bounds__(kNumWarpGroups* kNumWarpsPerGroup* WARP_SIZE, 1) void dispatch(
    void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayoutRange, int* outputCount,
    void* stagedRecv, int64_t* stagedRecvCountBuffer, void* stagedSend, const void* input, const int64_t* topkIdx,
    int* atomicCounterPerExpert, int* atomicFinishCounterPerExpert, int64_t* nextClean, int numNextCleanInt,
    int numTokens, int numMaxDispatchTokensPerRank, int numTopk, int hidden, int numExperts, int rank, int numRanks,
    int phases, void* rdmaBufferPtr, mscclpp::PortChannelDeviceHandle* portChannelHandles, void* const* peerRdmaBases,
    int ranksPerIpcDomain, DispatchLayout dispatchLayout) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto warpId = threadId / WARP_SIZE, laneId = get_lane_id();
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
        sharedNumTokensSentPerExpert, outputCount, stagedRecv, stagedRecvCountBuffer, stagedSend, input, topkIdx,
        atomicCounterPerExpert, atomicFinishCounterPerExpert, nextClean, numNextCleanInt, numTokens,
        numMaxDispatchTokensPerRank, numTopk, hidden, numExperts, rank, numRanks, rdmaBufferPtr, portChannelHandles,
        peerRdmaBases, ranksPerIpcDomain);
  }

  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // The recv phase consumes per-rank count signals and stagedRecv payloads
  // produced by the send phase in the same cooperative launch.
  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();

  dispatchRecv<kInputDType, kOutputDType, kNumWarpGroups, kNumWarpsPerGroup>(
      output, outputScales, outputSrcInfo, outputLayoutRange, outputCount, stagedRecv, stagedRecvCountBuffer,
      numMaxDispatchTokensPerRank, numExperts, numRanks, numLocalExperts, rank, numBytesPerMsg, hiddenInt4, hiddenBytes,
      numScales, dispatchLayout, warpGroupId, subWarpId, laneId, responsibleExpertIdx);
}

constexpr int kDispatchNumMaxTopK = 9;
constexpr int kDispatchNumWarpGroups = 3;
constexpr int kDispatchNumWarpsPerGroup = 10;

struct DispatchLaunchArgs {
  cudaLaunchConfig_t* launchConfig;
  void* output;
  float* outputScales;
  int* outputSrcInfo;
  int64_t* outputLayoutRange;
  int* outputCount;
  void* stagedRecv;
  int64_t* stagedRecvCountBuffer;
  void* stagedSend;
  const void* input;
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
  DispatchLayout dispatchLayout;
};

template <DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup>
void launchDispatchKernel(const DispatchLaunchArgs& args) {
  auto dispatchFunc = dispatch<DType::BF16, kOutputDType, kNumWarpGroups, kNumWarpsPerGroup>;
  CUDA_CHECK(cudaLaunchKernelEx(args.launchConfig, dispatchFunc, args.output, args.outputScales, args.outputSrcInfo,
                                args.outputLayoutRange, args.outputCount, args.stagedRecv, args.stagedRecvCountBuffer,
                                args.stagedSend, args.input, args.topkIdx, args.atomicCounterPerExpert,
                                args.atomicFinishCounterPerExpert, args.nextClean, args.numNextCleanInt, args.numTokens,
                                args.numMaxDispatchTokensPerRank, args.numTopk, args.hidden, args.numExperts, args.rank,
                                args.numRanks, args.phases, args.rdmaBufferPtr, args.portChannelHandles,
                                args.peerRdmaBases, args.ranksPerIpcDomain, args.dispatchLayout));
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
  auto stagedRecv = currentBuffer.recvDataBuffer_;
  auto stagedRecvCountBuffer = currentBuffer.recvCountBuffer_;
  auto stagedSend = currentBuffer.sendDataBuffer_;
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
  auto dispatchLayout = config.outputLayout_;
  auto phases = static_cast<int>(phase);
  auto rdmaBufferPtr = transport.rdmaBufferBase_;
  auto portChannelHandles = transport.portChannels_;
  auto peerRdmaBases = transport.peerBases_;
  auto ranksPerIpcDomain = transport.ranksPerIpcDomain_;
  static_assert(kDispatchNumMaxTopK + 1 <= kDispatchNumWarpGroups * kDispatchNumWarpsPerGroup,
                "Too many top-k selections");

  const auto numWarps = kDispatchNumWarpGroups * kDispatchNumWarpsPerGroup;
  const auto numSms = cell_div(numExperts, kDispatchNumWarpGroups);
  // dispatchSend divides payload warps into groups of numTopk sender warps.
  EP_HOST_ASSERT(numTopk > 0);
  EP_HOST_ASSERT(numTopk <= kDispatchNumMaxTopK);
  EP_HOST_ASSERT(dispatchLayout == DispatchLayout::EXPERT_MAJOR || dispatchLayout == DispatchLayout::FLAT);

  auto atomicCounterPerExpert = reinterpret_cast<int*>(workspace);
  auto atomicFinishCounterPerExpert = atomicCounterPerExpert + numExperts;
  EP_HOST_ASSERT(numExperts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(inputDType == DType::BF16);
  EP_HOST_ASSERT(hidden % 128 == 0);

  cudaLaunchConfig_t cfg = {dim3(numSms), dim3(numWarps * WARP_SIZE), 0, stream, nullptr, 0};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  DispatchLaunchArgs args{.launchConfig = &cfg,
                          .output = output,
                          .outputScales = outputScales,
                          .outputSrcInfo = outputSrcInfo,
                          .outputLayoutRange = outputLayout,
                          .outputCount = outputCount,
                          .stagedRecv = stagedRecv,
                          .stagedRecvCountBuffer = stagedRecvCountBuffer,
                          .stagedSend = stagedSend,
                          .input = input,
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
                          .ranksPerIpcDomain = ranksPerIpcDomain,
                          .dispatchLayout = dispatchLayout};

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
    static_assert(kInputDType == DType::F8E4M3, "Unsupported low-latency combine input dtype");
    const auto srcFp8 = reinterpret_cast<const __nv_fp8_storage_t*>(src);
    EP_DEVICE_ASSERT(scales != nullptr);

    for (int i = laneId; i < kHiddenBf16Int4; i += WARP_SIZE) {
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

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
MSCCLPP_DEVICE_INLINE void combineSend(void* stagedRecv, int64_t* stagedRecvFlagBuffer, void* stagedSend,
                                       const void* input, const float* inputScales, const int* srcInfo,
                                       const int64_t* layoutRange, int64_t* nextClean, int numNextCleanInt,
                                       int* atomicCleanFlag, int numMaxDispatchTokensPerRank, int numExperts, int rank,
                                       int numRanks, bool zeroCopy, void* rdmaBufferPtr,
                                       mscclpp::PortChannelDeviceHandle* portChannelHandles, void* const* peerRdmaBases,
                                       int ranksPerIpcDomain, int smId, int warpGroupId, int subWarpId, int laneId,
                                       int responsibleExpertIdx) {
  using InputType = typename DispatchDTypeTraits<kInputDType>::Type;
  using OutputType = typename DispatchDTypeTraits<kOutputDType>::Type;

  constexpr size_t numInputBytes = kHidden * sizeof(InputType);
  constexpr size_t numBytesPerSlot = kHidden * sizeof(OutputType);
  static_assert(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

  if (smId == 0 and warpGroupId == 0 and subWarpId == 0) {
    for (int i = laneId; i < numNextCleanInt; i += WARP_SIZE) nextClean[i] = 0;

    __syncwarp();
    if (laneId == 0) atomicAddReleaseDevice(atomicCleanFlag, numExperts);
  }

  if (responsibleExpertIdx < numExperts) {
    const auto numLocalExperts = numExperts / numRanks;
    const auto dstRank = responsibleExpertIdx / numLocalExperts;
    const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
    const auto globalExpertIdx = rank * numLocalExperts + localExpertIdx;
    const auto layout = __ldg(layoutRange + localExpertIdx * numRanks + dstRank);
    const int scaleStride = numRanks * numMaxDispatchTokensPerRank;
    const int numScales = kHidden / 128;
    const auto localInput = reinterpret_cast<const uint8_t*>(input) + localExpertIdx * scaleStride * numInputBytes;
    const auto localInputScales =
        inputScales == nullptr ? nullptr : inputScales + localExpertIdx * numScales * scaleStride;
    const auto localSrcInfo = srcInfo + localExpertIdx * numRanks * numMaxDispatchTokensPerRank;
    const auto stagedSendBase = reinterpret_cast<uint8_t*>(stagedSend) +
                                localExpertIdx * numRanks * numMaxDispatchTokensPerRank * numBytesPerSlot;

    int offset, numTokensToSend;
    unpack2(layout, numTokensToSend, offset);

    for (int tokenIdx = offset + subWarpId; tokenIdx < offset + numTokensToSend; tokenIdx += kNumWarpsPerGroup) {
      const auto inputRow = localInput + tokenIdx * numInputBytes;
      const auto inputScaleRow = localInputScales == nullptr ? nullptr : localInputScales + tokenIdx;
      const auto stagedSendMsg = reinterpret_cast<int*>(stagedSendBase + tokenIdx * numBytesPerSlot);
      const auto stagedSendRow = reinterpret_cast<uint8_t*>(stagedSendMsg);

      auto srcIdx = __ldg(localSrcInfo + tokenIdx);
      const auto stagedSendPtr = reinterpret_cast<int64_t>(stagedSendRow);
      const auto dstPtr = reinterpret_cast<uint64_t>(stagedRecv) +
                          (globalExpertIdx * numMaxDispatchTokensPerRank + srcIdx) * numBytesPerSlot;
      if (dstRank == rank) {
        const auto dstInt4Ptr = reinterpret_cast<int4*>(dstPtr);
        copyCombineInputToBf16<kInputDType, kHidden>(dstInt4Ptr, inputRow, inputScaleRow, scaleStride, laneId);
      } else {
        if (peerRdmaBases != nullptr && isIpcPeer(rank, dstRank, ranksPerIpcDomain)) {
          // Peer-mapped warp copy over NVLink. `zeroCopy` is irrelevant
          // on this path because we skip the rdma_send staging buffer.
          const auto peerDst = peerMappedPtrOf(dstPtr, peerRdmaBases, rdmaBufferPtr, dstRank);
          const auto peerDstInt4 = reinterpret_cast<int4*>(peerDst);
          copyCombineInputToBf16<kInputDType, kHidden>(peerDstInt4, inputRow, inputScaleRow, scaleStride, laneId);
        } else {
          const auto stagedSendInt4 = reinterpret_cast<int4*>(stagedSendPtr);
          if constexpr (kInputDType == DType::BF16) {
            if (not zeroCopy)
              copyCombineInputToBf16<kInputDType, kHidden>(stagedSendInt4, inputRow, inputScaleRow, scaleStride,
                                                           laneId);
          } else {
            copyCombineInputToBf16<kInputDType, kHidden>(stagedSendInt4, inputRow, inputScaleRow, scaleStride, laneId);
          }
          // MSCCL++ port-channel PUT.
          if (laneId == 0) {
            const auto dstOff = portChannelOffsetOf(dstPtr, rdmaBufferPtr);
            const auto srcOff = portChannelOffsetOf(static_cast<uint64_t>(stagedSendPtr), rdmaBufferPtr);
            portChannelHandles[localExpertIdx * numRanks + dstRank].put(dstOff, srcOff, kHidden * sizeof(OutputType));
          }
          __syncwarp();
        }
      }
    }

    static_assert(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
    asm volatile("bar.sync %0, %1;" ::"r"(warpGroupId + 1), "r"(kNumWarpsPerGroup * WARP_SIZE));
    if (subWarpId == 1 and laneId == 0) {
      while (ld_acquire_global(atomicCleanFlag) == 0);
      auto* flagPtr = stagedRecvFlagBuffer + globalExpertIdx;
      auto* portChannelHandle = dstRank == rank ? nullptr : portChannelHandles + localExpertIdx * numRanks + dstRank;
      publishSingleWriterSignal(flagPtr, static_cast<int64_t>(1), rank, dstRank, rdmaBufferPtr, portChannelHandle,
                                peerRdmaBases, ranksPerIpcDomain);
      atomicAddReleaseDevice(atomicCleanFlag, -1);
    }
    __syncwarp();
  }
}

template <DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
MSCCLPP_DEVICE_INLINE void combineRecv(void* output, void* stagedRecv, int64_t* stagedRecvFlagBuffer,
                                       const int64_t* topkIdx, const float* topkWeights, int numCombinedTokens,
                                       int numTopk, int numMaxDispatchTokensPerRank, int numExperts, int smId,
                                       int numSms, int threadId, int numThreads, int warpGroupId, int subWarpId,
                                       int laneId, int responsibleExpertIdx) {
  static_assert(kOutputDType == DType::BF16, "Only BF16 low-latency combine output is supported");
  using OutputType = typename DispatchDTypeTraits<kOutputDType>::Type;

  constexpr int kNumBf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr size_t hiddenBf16Int4 = kHidden / kNumBf16PerInt4;
  constexpr size_t numBytesPerSlot = kHidden * sizeof(OutputType);

  if (responsibleExpertIdx < numExperts) {
    static_assert(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
    if (subWarpId == 0 and laneId == 0) waitSignalNonZero(stagedRecvFlagBuffer + responsibleExpertIdx);
  }
  // All expert-result writes must be visible before owner ranks start the
  // weighted gather below.
  cg::this_grid().sync();

  EP_DEVICE_ASSERT(numTopk <= WARP_SIZE);
  static_assert(kHidden % (WARP_SIZE * kNumBf16PerInt4) == 0, "Invalid vectorization");
  for (size_t hiddenIdx = threadId; hiddenIdx < hiddenBf16Int4; hiddenIdx += numThreads) {
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
          auto stagedRecvType =
              reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(stagedRecv) +
                                           (regTopkIdx[i] * numMaxDispatchTokensPerRank + tokenIdx) * numBytesPerSlot);
          auto stagedRecvRow = reinterpret_cast<const uint8_t*>(stagedRecvType);

          auto stagedVec = ld_nc_global(reinterpret_cast<const int4*>(stagedRecvRow) + hiddenIdx);
          const auto stagedValues = reinterpret_cast<nv_bfloat16*>(&stagedVec);
#pragma unroll
          for (int j = 0; j < kNumBf16PerInt4; ++j)
            combinedValues[j] += static_cast<float>(stagedValues[j]) * regTopkWeights[i];
        }

      int4 combinedInt4;
      auto combinedOutput = reinterpret_cast<OutputType*>(&combinedInt4);
#pragma unroll
      for (int j = 0; j < kNumBf16PerInt4; ++j) combinedOutput[j] = static_cast<OutputType>(combinedValues[j]);
      (reinterpret_cast<int4*>(output) + tokenIdx * hiddenBf16Int4)[hiddenIdx] = combinedInt4;
    }
  }
}

template <DType kInputDType, DType kOutputDType, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden,
          int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups* kNumWarpsPerGroup* WARP_SIZE, 1) void combine(
    void* output, void* stagedRecv, int64_t* stagedRecvFlagBuffer, void* stagedSend, const void* input,
    const float* inputScales, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, int64_t* nextClean, int numNextCleanInt, int* atomicCleanFlag, int numCombinedTokens,
    int hidden, int numTopk, int numMaxDispatchTokensPerRank, int numExperts, int rank, int numRanks, int phases,
    bool zeroCopy, void* rdmaBufferPtr, mscclpp::PortChannelDeviceHandle* portChannelHandles,
    void* const* peerRdmaBases, int ranksPerIpcDomain) {
  const auto smId = static_cast<int>(blockIdx.x);
  const auto numSms = static_cast<int>(gridDim.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto numThreads = static_cast<int>(blockDim.x);
  const auto warpId = threadId / WARP_SIZE, laneId = get_lane_id();
  const auto warpGroupId = warpId / kNumWarpsPerGroup;
  const auto subWarpId = warpId % kNumWarpsPerGroup;
  const auto responsibleExpertIdx = smId * kNumWarpGroups + warpGroupId;

  if (phases & LOW_LATENCY_SEND_PHASE) {
    combineSend<kInputDType, kOutputDType, kNumWarpGroups, kNumWarpsPerGroup, kHidden>(
        stagedRecv, stagedRecvFlagBuffer, stagedSend, input, inputScales, srcInfo, layoutRange, nextClean,
        numNextCleanInt, atomicCleanFlag, numMaxDispatchTokensPerRank, numExperts, rank, numRanks, zeroCopy,
        rdmaBufferPtr, portChannelHandles, peerRdmaBases, ranksPerIpcDomain, smId, warpGroupId, subWarpId, laneId,
        responsibleExpertIdx);
  }

  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  combineRecv<kOutputDType, kNumWarpGroups, kNumWarpsPerGroup, kHidden, kNumMaxTopk>(
      output, stagedRecv, stagedRecvFlagBuffer, topkIdx, topkWeights, numCombinedTokens, numTopk,
      numMaxDispatchTokensPerRank, numExperts, smId, numSms, threadId, numThreads, warpGroupId, subWarpId, laneId,
      responsibleExpertIdx);
}

void combine(void* output, const void* input, const float* inputScales, const int64_t* topkIdx,
             const float* topkWeights, const int* srcInfo, const int64_t* layoutRange, const CombineConfig& config,
             const BufferSet& currentBuffer, const BufferSet& nextBuffer, const TransportContext& transport,
             void* workspace, cudaStream_t stream, Phase phase) {
  // Unpack configuration
  auto stagedRecv = currentBuffer.recvDataBuffer_;
  auto stagedRecvFlagBuffer = currentBuffer.recvCountBuffer_;
  auto stagedSend = currentBuffer.sendDataBuffer_;
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
    LAUNCH_KERNEL(&cfg, combineFunc, output, stagedRecv, stagedRecvFlagBuffer, stagedSend, input, inputScales,         \
                  topkIdx, topkWeights, srcInfo, layoutRange, nextClean, numNextCleanInt, atomicCleanFlag,             \
                  numCombinedTokens, hidden, numTopk, numMaxDispatchTokensPerRank, numExperts, rank, numRanks, phases, \
                  zeroCopy, rdmaBufferPtr, portChannelHandles, peerRdmaBases, ranksPerIpcDomain);                      \
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

  SETUP_LAUNCH_CONFIG(numSms, numWarps * WARP_SIZE, stream);
  SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
#undef COMBINE_LAUNCH
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
