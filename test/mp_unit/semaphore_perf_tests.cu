// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/atomic_device.hpp>
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

// ─── CUDA kernels: signal+wait ping-pong ──────────────────────────────────────

__constant__ mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle gSemaphorePerfTestHandle;

/// Old store-based signal: local fetch-add on outboundToken, then store to remote.
__device__ __forceinline__ void signalStore(mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle& sem) {
  auto outbound = sem.incOutbound();
  mscclpp::atomicStore(sem.remoteInboundToken, outbound, mscclpp::memoryOrderRelease);
}

// mode: 0 = new (signal()), 1 = old (store-based)
__global__ void kernelSemaphorePingPong(int rank, int nIters, int mode) {
  mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle& sem = gSemaphorePerfTestHandle;

  // Warmup
  for (int i = 0; i < 10; i++) {
    if ((rank ^ (i & 1)) == 0) {
      if (mode == 1)
        signalStore(sem);
      else
        sem.signal();
    } else {
      sem.wait();
    }
  }

  // Timed iterations — alternating signal/wait like the memory channel ping-pong
  for (int i = 0; i < nIters; i++) {
    if ((rank ^ (i & 1)) == 0) {
      if (mode == 1)
        signalStore(sem);
      else
        sem.signal();
    } else {
      sem.wait();
    }
  }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

static float runPingPong(std::shared_ptr<mscclpp::Communicator>& communicator, int rank, int nIters, int mode) {
  // Warmup
  kernelSemaphorePingPong<<<1, 1>>>(rank, nIters, mode);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  mscclpp::Timer timer;
  kernelSemaphorePingPong<<<1, 1>>>(rank, nIters, mode);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  return (float)timer.elapsed() / (float)nIters;
}

// ─── Test body ────────────────────────────────────────────────────────────────

PERF_TEST(SemaphorePerfTest, SignalPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  connectMesh(/*useIpc=*/true, /*useIb=*/false, /*useEthernet=*/false);

  int peerRank = (gEnv->rank == 0) ? 1 : 0;
  const int nIters = 1000;
  const std::string testName = ::mscclpp::test::currentTestName();

  // --- Old store-based signal (mode=1) ---
  auto semOld = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, connections[peerRank]);
  auto handleOld = semOld->deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gSemaphorePerfTestHandle, &handleOld, sizeof(handleOld)));
  float usOld = runPingPong(communicator, gEnv->rank, nIters, 1);

  // --- New red-based signal (mode=0) — need fresh semaphore ---
  auto semNew = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, connections[peerRank]);
  auto handleNew = semNew->deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gSemaphorePerfTestHandle, &handleNew, sizeof(handleNew)));
  float usNew = runPingPong(communicator, gEnv->rank, nIters, 0);

  if (gEnv->rank == 0) {
    float speedup = usOld / usNew;
    std::cout << testName << ":\n"
              << "  Store-based (old): " << std::setprecision(4) << usOld << " us/iter\n"
              << "  Red-based   (new): " << std::setprecision(4) << usNew << " us/iter\n"
              << "  Speedup:           " << std::setprecision(3) << speedup << "x\n";
  }
}
