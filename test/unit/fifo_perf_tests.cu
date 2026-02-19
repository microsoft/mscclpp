// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cassert>
#include <memory>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <unordered_map>

#include "../framework.hpp"

// Simple FIFO performance test to be run as part of unit_tests
// This is a performance test that can be excluded from coverage runs
// using the --exclude-perf-tests flag.

constexpr uint64_t TIMEOUT_SPINS = 1000000;
constexpr int MIN_TRIGGERS = 100;  // Reduced for faster unit test execution

__constant__ mscclpp::FifoDeviceHandle gFifoPerfDeviceHandle;

__global__ void kernelFifoPerfPush(size_t numTriggers) {
  mscclpp::FifoDeviceHandle& fifo = gFifoPerfDeviceHandle;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  mscclpp::ProxyTrigger trigger;
  for (size_t i = 1; i <= numTriggers; ++i) {
    trigger.fst = i;
    trigger.snd = tid ^ i;
    fifo.push(trigger);
  }
}

static bool consumePerfTriggers(std::unique_ptr<mscclpp::Fifo>& hostFifo, int numTriggers, int parallel) {
  int totalTriggers = numTriggers * parallel;
  std::unordered_map<int, int> triggerCounts;
  for (int i = 0; i < totalTriggers; ++i) {
    mscclpp::ProxyTrigger trigger;
    uint64_t spin = 0;
    do {
      trigger = hostFifo->poll();
      if (spin++ > TIMEOUT_SPINS) {
        return false;
      }
    } while (trigger.fst == 0 || trigger.snd == 0);

    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    trigger.snd = trigger.snd ^ trigger.fst;
    assert(triggerCounts[trigger.snd] + 1 == trigger.fst);
    triggerCounts[trigger.snd]++;
    hostFifo->pop();
  }
  return true;
}

PERF_TEST(FifoPerfTest, BasicPerformance) {
  int cudaDevice, numaNode;
  CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);

  const int fifoSize = 128;
  const int numTriggers = MIN_TRIGGERS;
  const int numParallel = 1;

  auto hostFifo = std::make_unique<mscclpp::Fifo>(fifoSize);
  mscclpp::FifoDeviceHandle hostHandle = hostFifo->deviceHandle();
  CUDA_CHECK(cudaMemcpyToSymbol(gFifoPerfDeviceHandle, &hostHandle, sizeof(mscclpp::FifoDeviceHandle)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Run kernel
  kernelFifoPerfPush<<<numParallel, 1, 0, stream>>>(numTriggers);
  CUDA_CHECK(cudaGetLastError());

  // Process triggers
  bool success = consumePerfTriggers(hostFifo, numTriggers, numParallel);
  ASSERT_TRUE(success);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
}
