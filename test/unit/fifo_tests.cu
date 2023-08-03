// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/utils.hpp>

#include "numa.hpp"

#define FLUSH_PERIOD (MSCCLPP_PROXY_FIFO_SIZE)  // should not exceed MSCCLPP_PROXY_FIFO_SIZE
#define ITER 10000                              // should be larger than MSCCLPP_PROXY_FIFO_SIZE for proper testing

__constant__ mscclpp::FifoDeviceHandle gFifoTestFifoDeviceHandle;
__global__ void kernelFifoTest() {
  if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;

  mscclpp::FifoDeviceHandle& fifo = gFifoTestFifoDeviceHandle;
  mscclpp::ProxyTrigger trigger;
  for (uint64_t i = 1; i < ITER + 1; ++i) {
    trigger.fst = i;
    trigger.snd = i;
    uint64_t curFifoHead = fifo.push(trigger);
    if (i % FLUSH_PERIOD == 0) {
      fifo.sync(curFifoHead);
    }
  }
}

TEST(FifoTest, Fifo) {
  ASSERT_LE(FLUSH_PERIOD, MSCCLPP_PROXY_FIFO_SIZE);

  int cudaNum;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaNum));
  int numaNode = mscclpp::getDeviceNumaNode(cudaNum);
  mscclpp::numaBind(numaNode);

  mscclpp::Fifo hostFifo;
  mscclpp::FifoDeviceHandle devFifo = hostFifo.deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gFifoTestFifoDeviceHandle, &devFifo, sizeof(devFifo)));

  kernelFifoTest<<<1, 1>>>();
  MSCCLPP_CUDATHROW(cudaGetLastError());

  mscclpp::ProxyTrigger trigger;
  trigger.fst = 0;
  trigger.snd = 0;

  uint64_t spin = 0;
  uint64_t flushCnt = 0;
  mscclpp::Timer timer(3);
  for (uint64_t i = 0; i < ITER; ++i) {
    while (trigger.fst == 0) {
      trigger = hostFifo.poll();

      if (spin++ > 1000000) {
        FAIL() << "Polling is stuck.";
      }
    }
    ASSERT_TRUE(trigger.fst == (i + 1));
    ASSERT_TRUE(trigger.snd == (i + 1));
    hostFifo.pop();
    if ((++flushCnt % FLUSH_PERIOD) == 0) {
      hostFifo.flushTail();
    }
    trigger.fst = 0;
    spin = 0;
  }
  hostFifo.flushTail(true);

  std::stringstream ss;
  ss << "FifoTest.Fifo: " << (float)timer.elapsed() / ITER << " us/iter\n";
  std::cout << ss.str();

  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}
