// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#if defined(TEST_GPUNETIO_CONSUMER)
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/port_channel.hpp>

#if TEST_EXPECT_GPUNETIO
#ifndef MSCCLPP_USE_GPUNETIO
#error "Installed ON library lost GPUNetIO usage requirements"
#endif
#else
#ifdef MSCCLPP_USE_GPUNETIO
#error "Installed OFF library unexpectedly enables GPUNetIO"
#endif
#endif

__global__ void compileConsumerChannel(mscclpp::PortChannelDeviceHandle channel) {
  channel.putWithSignal(0, 8, 8);
  channel.accumulate(16, 1);
  channel.flush();
}

void compileConsumerService(mscclpp::GpuNetIoService& service) {
  service.setup();
  service.setup(nullptr, 0);
  service.setup(std::vector<int>{}, 1);
  auto connection = service.connect(1, 0);
  auto memory = service.registerMemory(nullptr, 0);
  (void)service.exchangeMemory(connection, memory, 2);
  (void)service.buildSemaphore(connection, 3);
  (void)service.deviceContext();
  (void)service.deviceContext(1);
}

int main(int argc, char**) {
  for (const bool multiQp : {false, true}) {
    bool rejected = false;
    try {
      auto service = multiQp ? std::make_unique<mscclpp::GpuNetIoService>(nullptr, "test-device", 0, 2)
                             : std::make_unique<mscclpp::GpuNetIoService>(nullptr, "test-device", 0);
      if (argc > 1) compileConsumerService(*service);
    } catch (const mscclpp::Error& error) {
      rejected = error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
#if !TEST_EXPECT_GPUNETIO
      rejected = rejected && std::string(error.what()).find("built without GPUNetIO") != std::string::npos;
#endif
    }
    if (!rejected) return 2;
  }
  try {
    mscclpp::PortChannel channel(mscclpp::GpuNetIoSemaphore{}, {}, {});
  } catch (const mscclpp::Error&) {
    return 0;
  }
  return 1;
}

#elif defined(TEST_GPUNETIO_HEADER)
#if !defined(MSCCLPP_USE_GPUNETIO)
#error "GPUNetIO feature was not inherited from the linked library"
#endif
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
  channel.accumulate(64, 1);
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
