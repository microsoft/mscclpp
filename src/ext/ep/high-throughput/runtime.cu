// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// High-throughput runtime helpers.

#include "api.cuh"
#include "constants.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {
namespace detail {

template <int NumRanks>
__global__ void barrierKernel(int** taskFifoPtrs, int head, int rank) {
  barrier_device<NumRanks>(taskFifoPtrs, head, rank);
}

}  // namespace detail

void barrier(int** taskFifoPtrs, int head, int rank, int numRanks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                             \
  LAUNCH_KERNEL(&cfg, detail::barrierKernel<ranks>, taskFifoPtrs, head, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 32, stream);
  SWITCH_RANKS(numRanks, BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
