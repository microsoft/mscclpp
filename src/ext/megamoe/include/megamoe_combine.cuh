// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_COMBINE_CUH_
#define MSCCLPP_EXT_MEGAMOE_COMBINE_CUH_

#include "megamoe_device.cuh"
#include "megamoe_math.cuh"
#include "megamoe_quantization.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Widest slot batch. Only it loops over batches for larger topK, carrying FP32 sums between them.
constexpr int CombineSlots = 8;

// 16-byte vectors processed per thread per combine task; 8-slot batches use 1 vector to fit in 128 registers.
__host__ __device__ constexpr int combineItems(int slots) { return slots <= 2 ? 8 : (slots <= 4 ? 4 : 1); }

template <ActivationDtype ActDtype>
struct CombinePartialPolicy {
  using PackedType = uint4;

  MSCCLPP_DEVICE_INLINE static PackedType zero() { return make_uint4(0, 0, 0, 0); }

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static const PackedType* partialBase(const KernelParams& params) {
    return reinterpret_cast<const uint4*>(at<__bfloat16>(params.local, params.symmetric.partialOutput));
  }

  MSCCLPP_DEVICE_INLINE static mscclpp::f32x8 decodeToFloat8(const PackedType& rawValue) {
    mscclpp::bf16x8 packed;
    packed.storage = rawValue;
    return mscclpp::to<mscclpp::f32x8>(packed);
  }

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static void scaleOutputSum(mscclpp::f32x8&, const KernelParams&) {}
};

template <>
struct CombinePartialPolicy<ActivationDtype::Fp8Static> {
  using PackedType = uint2;

  MSCCLPP_DEVICE_INLINE static PackedType zero() { return make_uint2(0, 0); }

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static const PackedType* partialBase(const KernelParams& params) {
    return reinterpret_cast<const uint2*>(at<uint8_t>(params.local, params.symmetric.partialOutput));
  }

  MSCCLPP_DEVICE_INLINE static mscclpp::f32x8 decodeToFloat8(const PackedType& rawValue) {
    return dequantizeE4M3x8(rawValue);
  }

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static void scaleOutputSum(mscclpp::f32x8& sum, const KernelParams& params) {
    scaleFloat8(sum, params.config.activationScale);
  }
};

MSCCLPP_DEVICE_INLINE void storeCombined(__bfloat16* destination, const mscclpp::f32x8& sums, bool aligned) {
  auto values = mscclpp::to<mscclpp::bf16x8>(sums);
  if (aligned) {
    *reinterpret_cast<uint4*>(destination) = values.storage;
    return;
  }
#pragma unroll
  for (int element = 0; element < values.Size; ++element) destination[element] = values.data[element];
}

template <int Hidden, int Elements, int TaskVectors>
MSCCLPP_DEVICE_INLINE bool isValidCombineVector(int firstVec, int itemIdx, int nVectors) {
  return (Hidden > 0 && (Hidden / Elements) % TaskVectors == 0) || (firstVec + itemIdx * WARP_SIZE < nVectors);
}

// Prefetches the route IDs of the first slot batch one task ahead so partial loads never wait on route metadata.
template <int Slots>
MSCCLPP_DEVICE_INLINE void prefetchFirstBatchRouteIds(const int* routeIds, int taskIdx, int nTasks, int nTokenTasks,
                                                      int topK, int (&slotIds)[Slots]) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    slotIds[slot] = (taskIdx < nTasks && slot < topK) ? routeIds[(taskIdx / nTokenTasks) * topK + slot] : -1;
  }
}

template <int Slots, bool ProcessMultipleSlotBatches>
MSCCLPP_DEVICE_INLINE void selectRoutedSlots(const int* routeIds, int tokenIdx, int topK, int baseSlot,
                                             const int (&firstBatchIds)[Slots], bool (&routed)[Slots]) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    routed[slot] = (!ProcessMultipleSlotBatches || baseSlot == 0)
                       ? (firstBatchIds[slot] >= 0)
                       : (baseSlot + slot < topK && routeIds[tokenIdx * topK + baseSlot + slot] >= 0);
  }
}

// Issues all Slots * Items L2-only loads up front before FP32 decoding/accumulation.
template <class Policy, int Hidden, int Slots, int Items, int Elements, int TaskVectors>
MSCCLPP_DEVICE_INLINE void loadRoutedPartialBatch(const typename Policy::PackedType* sourceBase, int nVectors,
                                                  int firstVec, const bool (&routed)[Slots],
                                                  typename Policy::PackedType (&rawPartials)[Slots][Items]) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      rawPartials[slot][item] = Policy::zero();
      if (routed[slot] && isValidCombineVector<Hidden, Elements, TaskVectors>(firstVec, item, nVectors)) {
        rawPartials[slot][item] = __ldcg(sourceBase + slot * nVectors + item * WARP_SIZE);
      }
    }
  }
}

template <class Policy, bool CombineRouteWeights, int Slots, int Items>
MSCCLPP_DEVICE_INLINE void accumulatePartialBatch(const typename Policy::PackedType (&rawPartials)[Slots][Items],
                                                  const bool (&routed)[Slots], const float* slotWeightsBase,
                                                  mscclpp::f32x8 (&sums)[Items]) {
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const float slotWeight = (CombineRouteWeights && routed[slot]) ? slotWeightsBase[slot] : 1.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const mscclpp::f32x8 partialValues = Policy::decodeToFloat8(rawPartials[slot][item]);
      accumulateWeightedFloat8<CombineRouteWeights>(sums[item], partialValues, slotWeight);
    }
  }
}

template <class Policy, int Hidden, int Items, int Elements, int TaskVectors, class KernelParams>
MSCCLPP_DEVICE_INLINE void storeCombinedTaskItems(const KernelParams& params, __bfloat16* output, int tokenIdx,
                                                  int nVectors, int firstVec, bool aligned,
                                                  mscclpp::f32x8 (&sums)[Items]) {
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    Policy::scaleOutputSum(sums[item], params);
    if (isValidCombineVector<Hidden, Elements, TaskVectors>(firstVec, item, nVectors)) {
      storeCombined(output + (size_t(tokenIdx) * nVectors + firstVec + item * WARP_SIZE) * Elements, sums[item],
                    aligned);
    }
  }
}

// Sums the routed partial rows of every token in FP32 slot order and rounds once to BF16. A warp task covers Items
// consecutive warp-wide vectors of one token row; each lane issues all Slots * Items loads before accumulating.
template <int Hidden, int Slots, bool ProcessMultipleSlotBatches = false, bool CombineRouteWeights = false,
          class KernelParams>
MSCCLPP_DEVICE_INLINE void combineSlots(const KernelParams& params, int nTokens, __bfloat16* output) {
  using Policy = CombinePartialPolicy<KernelParams::Activation>;
  using PackedPartial = typename Policy::PackedType;
  constexpr int Elements = mscclpp::bf16x8::Size;
  constexpr int Items = combineItems(Slots);
  constexpr int TaskVectors = Items * WARP_SIZE;
  static_assert((Hidden == 0 || Hidden % Elements == 0) && Slots <= CombineSlots);

  const int nVectors = (Hidden > 0 ? Hidden : params.config.hidden) / Elements;
  const int nTokenTasks = (nVectors + TaskVectors - 1) / TaskVectors;
  const int nTasks = nTokens * nTokenTasks;
  const int topK = params.config.topK;
  const PackedPartial* partialBase = Policy::partialBase(params);
  const int* routeIds = at<int>(params.local, params.symmetric.topkIds);
  const float* routeWeights = at<float>(params.local, params.symmetric.topkWeights);
  const bool aligned = (reinterpret_cast<uintptr_t>(output) & 15) == 0;
  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int nWarps = gridDim.x * (KernelParams::ThreadCount / WARP_SIZE);

  int taskIdx = (threadIdx.x / WARP_SIZE) * int(gridDim.x) + int(blockIdx.x);
  int nextBatchIds[Slots];
  prefetchFirstBatchRouteIds<Slots>(routeIds, taskIdx, nTasks, nTokenTasks, topK, nextBatchIds);

  for (; taskIdx < nTasks; taskIdx += nWarps) {
    const int tokenIdx = taskIdx / nTokenTasks;
    const int firstVec = (taskIdx % nTokenTasks) * TaskVectors + laneIdx;
    int firstBatchIds[Slots];
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) firstBatchIds[slot] = nextBatchIds[slot];
    prefetchFirstBatchRouteIds<Slots>(routeIds, taskIdx + nWarps, nTasks, nTokenTasks, topK, nextBatchIds);

    mscclpp::f32x8 sums[Items];
#pragma unroll
    for (int item = 0; item < Items; ++item) zeroFloat8(sums[item]);

    for (int baseSlot = 0; baseSlot < (ProcessMultipleSlotBatches ? topK : 1); baseSlot += Slots) {
      bool routed[Slots];
      selectRoutedSlots<Slots, ProcessMultipleSlotBatches>(routeIds, tokenIdx, topK, baseSlot, firstBatchIds, routed);

      const PackedPartial* sourceBase = partialBase + size_t(tokenIdx * topK + baseSlot) * nVectors + firstVec;
      PackedPartial rawPartials[Slots][Items];
      loadRoutedPartialBatch<Policy, Hidden, Slots, Items, Elements, TaskVectors>(sourceBase, nVectors, firstVec,
                                                                                  routed, rawPartials);
      accumulatePartialBatch<Policy, CombineRouteWeights, Slots, Items>(
          rawPartials, routed, routeWeights + tokenIdx * topK + baseSlot, sums);
    }

    storeCombinedTaskItems<Policy, Hidden, Items, Elements, TaskVectors>(params, output, tokenIdx, nVectors, firstVec,
                                                                         aligned, sums);
  }
}

template <int Hidden = 0, class KernelParams>
MSCCLPP_DEVICE_INLINE bool hasCombineWork(const KernelParams& params, int nTokens) {
  const int vectors = (Hidden > 0 ? Hidden : params.config.hidden) / mscclpp::bf16x8::Size;
  const int topK = params.config.topK;
  const int slots = topK == 1 ? 1 : (topK == 2 ? 2 : (topK <= 4 ? 4 : CombineSlots));
  const int items = combineItems(slots);
  const int tokenTasks = (vectors + items * WARP_SIZE - 1) / (items * WARP_SIZE);
  return int(blockIdx.x) < nTokens * tokenTasks;
}

template <int Hidden = 0, bool CombineRouteWeights = false, class KernelParams>
MSCCLPP_DEVICE_INLINE void combineResultsMode(const KernelParams& params, int nTokens, __bfloat16* output) {
  const int topK = params.config.topK;
  if (topK == 1)
    combineSlots<Hidden, 1, false, CombineRouteWeights>(params, nTokens, output);
  else if (topK == 2)
    combineSlots<Hidden, 2, false, CombineRouteWeights>(params, nTokens, output);
  else if (topK <= 4)
    combineSlots<Hidden, 4, false, CombineRouteWeights>(params, nTokens, output);
  else if (topK <= CombineSlots)
    combineSlots<Hidden, CombineSlots, false, CombineRouteWeights>(params, nTokens, output);
  else
    combineSlots<Hidden, CombineSlots, true, CombineRouteWeights>(params, nTokens, output);
}

template <int Hidden = 0, class KernelParams>
MSCCLPP_DEVICE_INLINE void combineResults(const KernelParams& params, int nTokens, __bfloat16* output) {
  if (params.config.combineRouteWeights)
    combineResultsMode<Hidden, true>(params, nTokens, output);
  else
    combineResultsMode<Hidden, false>(params, nTokens, output);
}

template <int Hidden = 0, class KernelParams>
__device__ void joinAndCombineOutputs(const KernelParams& params, int nTokens, __bfloat16* output) {
  if (params.config.computeOnly) return;
  if constexpr (KernelParams::WeightMxfp4 && KernelParams::Activation == ActivationDtype::Mxfp8) {
    if (!params.config.combineRouteWeights && params.config.worldSize == 1 && params.config.topK == 1) return;
  }
  if (!syncPeerPartialOutputs(params, hasCombineWork<Hidden>(params, nTokens))) return;
  traceKernel(KernelTracePhase::Combine, true);
  combineResults<Hidden>(params, nTokens, output);
  traceKernel(KernelTracePhase::Combine, false);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_COMBINE_CUH_
