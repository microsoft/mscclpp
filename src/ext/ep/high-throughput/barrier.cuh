// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <mscclpp/memory_channel_device.hpp>

#include "device_helpers.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {

template <int NumRanks>
__forceinline__ __device__ void barrier_device(mscclpp::BaseMemoryChannelDeviceHandle* channels, int rank) {
#ifdef MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT
  constexpr int64_t MaxSpinCount = 10000000;
#else
  constexpr int64_t MaxSpinCount = 100000000;
#endif
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

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
