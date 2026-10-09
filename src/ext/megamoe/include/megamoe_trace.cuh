// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TRACE_CUH_
#define MSCCLPP_EXT_MEGAMOE_TRACE_CUH_

#include <cstddef>
#include <cstdint>

#include "megamoe_specialization.hpp"

#ifndef MSCCLPP_MEGAMOE_TRACE
#if defined(MSCCLPP_MEGAMOE_W4_TRACE)
#define MSCCLPP_MEGAMOE_TRACE MSCCLPP_MEGAMOE_W4_TRACE
#else
#define MSCCLPP_MEGAMOE_TRACE 0
#endif
#endif

#if MSCCLPP_MEGAMOE_TRACE
#include <cuda_runtime.h>
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

enum class KernelTracePhase : uint32_t {
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
  InputReadyAtEntry,
  TransformFc1,
  TransformFc2,
  Activation,
  Fc1Store
};

#if MSCCLPP_MEGAMOE_TRACE
constexpr int KernelTraceCtas = 2;
constexpr int KernelTraceWarpsPerCta = 16;
constexpr int KernelTraceTracks = KernelTraceCtas * KernelTraceWarpsPerCta;
constexpr int KernelTraceCapacity = 8192;
struct KernelTraceEvent {
  uint64_t timestamp;
  uint32_t code;
  int32_t payload;
};
// Each kernel TU owns its recorder and its matching reset/copy exports.
static __device__ bool kernelTraceEnabled = false;
static __device__ uint32_t kernelTraceCounts[KernelTraceTracks];
static __device__ KernelTraceEvent kernelTraceEvents[KernelTraceTracks * KernelTraceCapacity];

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
static inline cudaError_t resetKernelTrace() {
  void* counts = nullptr;
  auto result = cudaGetSymbolAddress(&counts, kernelTraceCounts);
  if (result != cudaSuccess) return result;
  result = cudaMemset(counts, 0, sizeof(kernelTraceCounts));
  if (result != cudaSuccess) return result;
  bool enabled = true;
  return cudaMemcpyToSymbol(kernelTraceEnabled, &enabled, sizeof(enabled));
}

static inline cudaError_t copyKernelTrace(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  if (!events || !counts || bytes < sizeof(kernelTraceEvents) || countBytes < sizeof(kernelTraceCounts))
    return cudaErrorInvalidValue;
  bool enabled = false;
  auto result = cudaMemcpyToSymbol(kernelTraceEnabled, &enabled, sizeof(enabled));
  if (result != cudaSuccess) return result;
  result = cudaMemcpyFromSymbol(counts, kernelTraceCounts, sizeof(kernelTraceCounts));
  if (result != cudaSuccess) return result;
  return cudaMemcpyFromSymbol(events, kernelTraceEvents, sizeof(kernelTraceEvents));
}
#endif
#endif

__device__ __forceinline__ void traceKernel(KernelTracePhase phase, bool begin, int32_t payload = 0) {
#if MSCCLPP_MEGAMOE_TRACE
  if (blockIdx.x < KernelTraceCtas && threadIdx.x / warpSize < KernelTraceWarpsPerCta && threadIdx.x % warpSize == 0 &&
      kernelTraceEnabled) {
    int track = blockIdx.x * KernelTraceWarpsPerCta + threadIdx.x / warpSize;
    uint32_t index = kernelTraceCounts[track]++;
    if (index < KernelTraceCapacity) {
      uint64_t timestamp;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) : : "memory");
      kernelTraceEvents[track * KernelTraceCapacity + index] = {timestamp, uint32_t(phase) * 2 + !begin, payload};
    }
  }
#endif
}

__device__ __forceinline__ uint64_t traceKernelClock() {
#if MSCCLPP_MEGAMOE_TRACE == 1
  uint64_t timestamp;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) : : "memory");
  return timestamp;
#else
  return 0;
#endif
}

__device__ __forceinline__ void traceKernelTotal(KernelTracePhase phase, uint64_t nanoseconds, int count) {
#if MSCCLPP_MEGAMOE_TRACE == 1
  if (blockIdx.x < KernelTraceCtas && threadIdx.x / warpSize < KernelTraceWarpsPerCta && threadIdx.x % warpSize == 0 &&
      kernelTraceEnabled) {
    int track = blockIdx.x * KernelTraceWarpsPerCta + threadIdx.x / warpSize;
    uint32_t index = kernelTraceCounts[track]++;
    if (index < KernelTraceCapacity)
      kernelTraceEvents[track * KernelTraceCapacity + index] = {nanoseconds, 0x80000000u | uint32_t(phase), count};
  }
#endif
}

__device__ __forceinline__ void traceKernelCount(KernelTracePhase phase, uint32_t ready, int attempts) {
#if MSCCLPP_MEGAMOE_TRACE == 1
  if (blockIdx.x < KernelTraceCtas && threadIdx.x / warpSize < KernelTraceWarpsPerCta && threadIdx.x % warpSize == 0 &&
      kernelTraceEnabled) {
    int track = blockIdx.x * KernelTraceWarpsPerCta + threadIdx.x / warpSize;
    uint32_t index = kernelTraceCounts[track]++;
    if (index < KernelTraceCapacity)
      kernelTraceEvents[track * KernelTraceCapacity + index] = {ready, 0x20000000u | uint32_t(phase), attempts};
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TRACE_CUH_
