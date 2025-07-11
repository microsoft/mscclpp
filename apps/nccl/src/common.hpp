// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NCCL_COMMON_HPP_
#define NCCL_COMMON_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/env.hpp>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#else
#define WARP_SIZE 32
#endif

constexpr int NUM_NVLS_CONNECTION = 8;
constexpr int NUM_SEMAPHORES = 64;

constexpr int MAX_NRANKS_PER_NODE = 8;

constexpr int SCRATCH_SIZE = 2 * 1024 * 1024 * 70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB
static bool mscclppDisableChannelCache = mscclpp::env()->disableChannelCache;

__device__ mscclpp::DeviceSyncer deviceSyncer;
__constant__ mscclpp::DeviceSemaphore deviceSemaphore[NUM_SEMAPHORES];

#endif  // NCCL_COMMON_HPP_
