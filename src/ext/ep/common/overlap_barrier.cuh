// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMMON_OVERLAP_BARRIER_CUH_
#define MSCCLPP_EP_COMMON_OVERLAP_BARRIER_CUH_

#include <mscclpp/memory_channel_device.hpp>

#include "device_helpers.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {
namespace common {

template <int NumRanks>
__forceinline__ __device__ void overlapBarrier(mscclpp::BaseMemoryChannelDeviceHandle* channels, int rank) {
  constexpr int64_t MaxSpinCount = 100'000'000;
  const int laneId = static_cast<int>(threadIdx.x) % WARP_SIZE;
  EP_DEVICE_ASSERT(NumRanks <= WARP_SIZE);

  if (laneId < NumRanks && laneId != rank) {
    channels[laneId].signal();
  }
  __syncwarp();
  if (laneId < NumRanks && laneId != rank) {
    channels[laneId].wait(MaxSpinCount);
  }
  __syncwarp();
}

}  // namespace common
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_OVERLAP_BARRIER_CUH_
