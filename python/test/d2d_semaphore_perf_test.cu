// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/semaphore_device.hpp>

using mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle;

/// Store-based signal (old path): local fetch-add on outboundToken, then store to remote.
__device__ __forceinline__ void signalStore(MemoryDevice2DeviceSemaphoreDeviceHandle& sem) {
  auto outbound = sem.incOutbound();
  mscclpp::atomicStore(sem.remoteInboundToken, outbound, mscclpp::memoryOrderRelease);
}

/// Red-based signal (new path): single fire-and-forget atomic add on remote.
__device__ __forceinline__ void signalRed(MemoryDevice2DeviceSemaphoreDeviceHandle& sem) {
  mscclpp::atomicAdd(sem.remoteInboundToken, (uint64_t)1, mscclpp::memoryOrderRelease);
}

extern "C" __global__ void __launch_bounds__(1024, 1)
    d2d_semaphore_perf(MemoryDevice2DeviceSemaphoreDeviceHandle* semaphores, int myRank, int nRanks, int nIters,
                       int useRed, long long* elapsedOut) {
  int tid = threadIdx.x;
  if (tid < nRanks && tid != myRank) {
    // Warmup
    for (int i = 0; i < 10; i++) {
      if (useRed)
        signalRed(semaphores[tid]);
      else
        signalStore(semaphores[tid]);
      semaphores[tid].wait();
    }

    // Timed iterations
    long long start = clock64();
    for (int i = 0; i < nIters; i++) {
      if (useRed)
        signalRed(semaphores[tid]);
      else
        signalStore(semaphores[tid]);
      semaphores[tid].wait();
    }
    long long end = clock64();
    elapsedOut[tid] = end - start;
  }
}
