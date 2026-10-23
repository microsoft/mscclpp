// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROLES_CUH_

#include "megamoe_collective.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule, class ClusterShape>
__device__ __forceinline__ Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count = cute::size(ClusterShape{}) * (Schedule::EpilogueEnd - Schedule::EpilogueBegin) * 32;
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params, ClusterShape{}, cute::true_type{}, cute::false_type{});
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
