// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// HT runtime helpers. This file contains the device barrier launcher
// used to coordinate direct peer access to the symmetric HT buffers.

#include "constants.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                 \
  LAUNCH_KERNEL(&cfg, barrier<ranks>, task_fifo_ptrs, head, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 32, stream);
  SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode
}  // namespace ep
}  // namespace mscclpp
