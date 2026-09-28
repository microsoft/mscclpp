// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/utils.hpp>

#include "../framework.hpp"
#include "utils_internal.hpp"

#define ITER 10000  // should be larger than the FIFO size for proper testing

__constant__ mscclpp::FifoDeviceHandle gFifoTestFifoDeviceHandle;
__global__ void kernelFifoTest() {
  if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;

  mscclpp::FifoDeviceHandle& fifo = gFifoTestFifoDeviceHandle;
  mscclpp::ProxyTrigger trigger;
  // Payloads start at 0: no trigger value is reserved by the FIFO any more.
  for (uint64_t i = 0; i < ITER; ++i) {
    trigger.fst = i;
    trigger.snd = i;
    uint64_t curFifoHead = fifo.push(trigger);
    if (i % fifo.size == 0) {
      fifo.sync(curFifoHead);
    }
  }
}

TEST(FifoTest, Fifo) {
  int cudaNum;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaNum));
  int numaNode = mscclpp::getDeviceNumaNode(cudaNum);
  mscclpp::numaBind(numaNode);

  mscclpp::Fifo hostFifo;
  if (hostFifo.size() >= ITER) {
    FAIL() << "ITER is too small for proper testing.";
  }

  mscclpp::FifoDeviceHandle devFifo = hostFifo.deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gFifoTestFifoDeviceHandle, &devFifo, sizeof(devFifo)));

  kernelFifoTest<<<1, 1>>>();
  MSCCLPP_CUDATHROW(cudaGetLastError());

  mscclpp::ProxyTrigger trigger;
  mscclpp::Timer timer(3);
  for (uint64_t i = 0; i < ITER; ++i) {
    mscclpp::Timer pollTimer;
    while (!hostFifo.poll(trigger)) {
      if (pollTimer.elapsed() > 5'000'000) {
        FAIL() << "Polling timed out at trigger " << i;
      }
    }
    ASSERT_TRUE(trigger.fst == i);
    ASSERT_TRUE(trigger.snd == i);
    hostFifo.pop();
  }

  std::stringstream ss;
  ss << "FifoTest.Fifo: " << (float)timer.elapsed() / ITER << " us/iter\n";
  std::cout << ss.str();

  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}

__constant__ mscclpp::FifoDeviceHandle gFifoZeroTestHandle;

// A trigger whose words are both zero must round-trip. Under the old protocol a zero first word
// meant "not yet written", so this trigger was invisible and the FIFO stalled on its slot.
__global__ void kernelFifoZeroTrigger(int count) {
  if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
  mscclpp::FifoDeviceHandle& fifo = gFifoZeroTestHandle;
  for (int i = 0; i < count; ++i) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 0;
    trigger.snd = 0;
    fifo.push(trigger);
  }
}

TEST(FifoTest, ZeroTrigger) {
  const int count = 32;
  mscclpp::Fifo hostFifo;
  mscclpp::FifoDeviceHandle devFifo = hostFifo.deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gFifoZeroTestHandle, &devFifo, sizeof(devFifo)));

  kernelFifoZeroTrigger<<<1, 1>>>(count);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  // count is below the FIFO capacity, so the producer cannot wait for the consumer here.
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  mscclpp::ProxyTrigger trigger;
  for (int i = 0; i < count; ++i) {
    uint64_t spin = 0;
    while (!hostFifo.poll(trigger)) {
      if (spin++ > 1000000) {
        FAIL() << "Polling is stuck on a zero-valued trigger " << i;
      }
    }
    ASSERT_TRUE(trigger.fst == 0);
    ASSERT_TRUE(trigger.snd == 0);
    hostFifo.pop();
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}

__constant__ mscclpp::FifoDeviceHandle gFifoWrapTestHandle;

// Push exactly a whole number of laps. A parity polarity error shows up here: the consumer either
// stalls at a lap boundary or accepts the previous lap's trigger a second time.
__global__ void kernelFifoWrap(int laps, int fifoSize) {
  if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
  mscclpp::FifoDeviceHandle& fifo = gFifoWrapTestHandle;
  for (int i = 0; i < laps * fifoSize; ++i) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = uint64_t(i);
    trigger.snd = ~uint64_t(i);
    trigger.fields.reserved = 0;  // the FIFO owns this bit
    uint64_t head = fifo.push(trigger);
    if ((i + 1) % fifoSize == 0) fifo.sync(head);
  }
}

TEST(FifoTest, WrapAtLapBoundary) {
  const int laps = 4;
  mscclpp::Fifo hostFifo;
  const int fifoSize = hostFifo.size();
  mscclpp::FifoDeviceHandle devFifo = hostFifo.deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gFifoWrapTestHandle, &devFifo, sizeof(devFifo)));

  kernelFifoWrap<<<1, 1>>>(laps, fifoSize);
  MSCCLPP_CUDATHROW(cudaGetLastError());

  mscclpp::ProxyTrigger trigger;
  for (int i = 0; i < laps * fifoSize; ++i) {
    uint64_t spin = 0;
    while (!hostFifo.poll(trigger)) {
      if (spin++ > 10000000) {
        FAIL() << "Polling is stuck at position " << i << " (lap " << i / fifoSize << ")";
      }
    }
    ASSERT_TRUE(trigger.fst == uint64_t(i));
    mscclpp::ProxyTrigger expected;
    expected.snd = ~uint64_t(i);
    expected.fields.reserved = 0;
    ASSERT_TRUE(trigger.snd == expected.snd);
    hostFifo.pop();
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
}

TEST(FifoTest, RejectsNonPowerOfTwoSize) {
  try {
    mscclpp::Fifo fifo(500);
    FAIL() << "Expected a non-power-of-two FIFO size to throw";
  } catch (const mscclpp::Error& e) {
    EXPECT_EQ(e.getErrorCode(), mscclpp::ErrorCode::InvalidUsage);
  }
}
