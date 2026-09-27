// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TRACE_CUH_
#define MSCCLPP_EXT_MEGAMOE_TRACE_CUH_

#include <cstdint>

#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

enum class W4TracePhase : uint32_t {
  Routing,
  Dispatch,
  TokenReady,
  LoadFc1,
  LoadFc2,
  LoadAcquire,
  TmaIssue,
  MmaFc1,
  MmaFc2,
  MmaInputWait,
  AccumulatorFree,
  MmaIssue,
  EpilogueReady,
  ActivationQuantize,
  Fc2Store,
  OutputJoin,
  Combine,
  StageRelease,
  MmaWeightWait,
  MmaActivationWait,
  WeightReadyAtEntry,
  ActivationReadyAtEntry,
  InputReadyAtEntry
};

#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE
constexpr int W4TraceWarps = 2 * 16;
constexpr int W4TraceCapacity = 8192;
struct W4TraceEvent {
  uint64_t timestamp;
  uint32_t code;
  int32_t payload;
};
static __device__ bool w4TraceEnabled = false;
static __device__ uint32_t w4TraceCounts[W4TraceWarps];
static __device__ W4TraceEvent w4TraceEvents[W4TraceWarps * W4TraceCapacity];
#endif

__device__ __forceinline__ void traceW4(W4TracePhase phase, bool begin, int32_t payload = 0) {
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE
  if (blockIdx.x < 2 && threadIdx.x % 32 == 0 && w4TraceEnabled) {
    int track = blockIdx.x * 16 + threadIdx.x / 32;
    uint32_t index = w4TraceCounts[track]++;
    if (index < W4TraceCapacity) {
      uint64_t timestamp;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) : : "memory");
      w4TraceEvents[track * W4TraceCapacity + index] = {timestamp, uint32_t(phase) * 2 + !begin, payload};
    }
  }
#endif
}

__device__ __forceinline__ uint64_t traceW4Clock() {
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE == 1
  uint64_t timestamp;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) : : "memory");
  return timestamp;
#else
  return 0;
#endif
}

__device__ __forceinline__ void traceW4Total(W4TracePhase phase, uint64_t nanoseconds, int count) {
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE == 1
  if (blockIdx.x < 2 && threadIdx.x % 32 == 0 && w4TraceEnabled) {
    int track = blockIdx.x * 16 + threadIdx.x / 32;
    uint32_t index = w4TraceCounts[track]++;
    if (index < W4TraceCapacity)
      w4TraceEvents[track * W4TraceCapacity + index] = {nanoseconds, 0x80000000u | uint32_t(phase), count};
  }
#endif
}

__device__ __forceinline__ void traceW4Count(W4TracePhase phase, uint32_t ready, int attempts) {
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE == 1
  if (blockIdx.x < 2 && threadIdx.x % 32 == 0 && w4TraceEnabled) {
    int track = blockIdx.x * 16 + threadIdx.x / 32;
    uint32_t index = w4TraceCounts[track]++;
    if (index < W4TraceCapacity)
      w4TraceEvents[track * W4TraceCapacity + index] = {ready, 0x20000000u | uint32_t(phase), attempts};
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TRACE_CUH_
