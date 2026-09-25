// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#if defined(TEST_GPUNETIO_HEADER)
#if defined(TEST_EXISTING_UNROLL_MACROS)
#define DO_PRAGMA(argument) 71
#define NVCC_PRAGMA_UNROLL(count) 73
#define NVCC_PRAGMA_UNROLL_AUTO 79
#define NVCC_PRAGMA_UNROLL_DISABLED 83
#else
#if defined(DO_PRAGMA) || defined(NVCC_PRAGMA_UNROLL) || defined(NVCC_PRAGMA_UNROLL_AUTO) || \
    defined(NVCC_PRAGMA_UNROLL_DISABLED)
#error "Header hygiene test requires initially undefined unroll macros"
#endif
#endif

#include <mscclpp/port_channel_device.hpp>

#if defined(TEST_EXISTING_UNROLL_MACROS)
static_assert(DO_PRAGMA(unused) == 71);
static_assert(NVCC_PRAGMA_UNROLL(4) == 73);
static_assert(NVCC_PRAGMA_UNROLL_AUTO == 79);
static_assert(NVCC_PRAGMA_UNROLL_DISABLED == 83);
#else
#if defined(DO_PRAGMA) || defined(NVCC_PRAGMA_UNROLL) || defined(NVCC_PRAGMA_UNROLL_AUTO) || \
    defined(NVCC_PRAGMA_UNROLL_DISABLED)
#error "GPUNetIO header leaked unroll compatibility macros"
#endif
#endif

__global__ void compileGpuNetIoHeader(mscclpp::PortChannelDeviceHandle channel) {
  channel.put(0, 16, 8);
  channel.signal();
  channel.putWithSignal(0, 16, 8);
  channel.atomicAdd(64, 1);
  channel.flush();
}

#else
#include "../framework.hpp"

#undef NDEBUG
#ifndef DEBUG_BUILD
#define DEBUG_BUILD
#endif  // DEBUG_BUILD
#include <assert.h>

#include <mscclpp/poll_device.hpp>

TEST(CompileTest, Assert) { assert(true); }
#endif
