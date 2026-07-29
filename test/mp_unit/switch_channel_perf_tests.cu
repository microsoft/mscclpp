// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/switch_channel_device.hpp>
#include <ranges>

#include "mp_unit_tests.hpp"

void SwitchChannelPerfTest::SetUp() {
  // Need at least two ranks within a node, and NVLS (switch multimem) support.
  if (gEnv->nRanksPerNode < 2) {
    SKIP_TEST();
  }
  if (!mscclpp::isNvlsSupported()) {
    SKIP_TEST();
  }
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void SwitchChannelPerfTest::TearDown() { CommunicatorTestBase::TearDown(); }

__constant__ mscclpp::SwitchChannelDeviceHandle gPerfSwitchChan;

// Back-to-back ordered barriers; one thread per rank drives the group barrier.
__global__ void kernelSwitchBarrierLatency(int nIters) {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  for (int i = 0; i < nIters; i++) {
    gPerfSwitchChan.signal();
    gPerfSwitchChan.wait();
  }
#endif  // (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
}

PERF_TEST(SwitchChannelPerfTest, BarrierLatency) {
  if (gEnv->rank >= numRanksToUse) return;

  auto rankView = std::views::iota(0, numRanksToUse);
  std::vector<int> ranks(rankView.begin(), rankView.end());

  const size_t bufSize = 1024;
  auto buffer = mscclpp::GpuBuffer<float>(bufSize / sizeof(float));

  auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, bufSize);
  auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), bufSize);
  auto deviceHandle = switchChannel.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gPerfSwitchChan, &deviceHandle, sizeof(deviceHandle)));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  const int nIters = 1000;

  communicator->bootstrap()->barrier();

  // Warmup run
  kernelSwitchBarrierLatency<<<1, 1>>>(nIters);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  // Timed run
  mscclpp::Timer timer;
  kernelSwitchBarrierLatency<<<1, 1>>>(nIters);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    ::mscclpp::test::reportPerfResult("latency", (float)timer.elapsed() / (float)nIters, "us/iter");
  }
}