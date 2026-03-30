// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/semaphore.hpp>

#include "mp_unit_tests.hpp"

void SemaphorePerfTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    SKIP_TEST();
  }
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void SemaphorePerfTest::TearDown() { CommunicatorTestBase::TearDown(); }

// ─── CUDA kernel: signal+wait ping-pong ───────────────────────────────────────

__constant__ mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle gSemaphorePerfTestHandle;

__global__ void kernelSemaphorePingPong(int rank, int nIters) {
  mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle& sem = gSemaphorePerfTestHandle;

  // Warmup
  for (int i = 0; i < 10; i++) {
    if ((rank ^ (i & 1)) == 0) {
      sem.signal();
    } else {
      sem.wait();
    }
  }

  // Timed iterations — alternating signal/wait like the memory channel ping-pong
  for (int i = 0; i < nIters; i++) {
    if ((rank ^ (i & 1)) == 0) {
      sem.signal();
    } else {
      sem.wait();
    }
  }
}

// ─── Test body ────────────────────────────────────────────────────────────────

PERF_TEST(SemaphorePerfTest, SignalPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  connectMesh(/*useIpc=*/true, /*useIb=*/false, /*useEthernet=*/false);

  int peerRank = (gEnv->rank == 0) ? 1 : 0;
  auto d2dSemaphore = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, connections[peerRank]);

  auto devHandle = d2dSemaphore->deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gSemaphorePerfTestHandle, &devHandle, sizeof(devHandle)));

  const int nIters = 1000;
  const std::string testName = ::mscclpp::test::currentTestName();

  // Warmup run
  kernelSemaphorePingPong<<<1, 1>>>(gEnv->rank, nIters);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // Timed run
  mscclpp::Timer timer;
  kernelSemaphorePingPong<<<1, 1>>>(gEnv->rank, nIters);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nIters << " us/iter\n";
  }
}
