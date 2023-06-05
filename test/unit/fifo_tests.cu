#include <gtest/gtest.h>
#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/fifo.hpp>

#define ITER 1000000

__const__ mscclpp::DeviceProxyFifo gFifoTestDeviceProxyFifo;
__global__ void kernelFifoTest()
{
  if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;

  mscclpp::DeviceProxyFifo& fifo = gFifoTestDeviceProxyFifo;
  mscclpp::ProxyTrigger trigger;
  for (uint64_t i = 1; i < ITER + 1; ++i) {
    trigger.fst = i;
    trigger.snd = i;
    fifo.push(trigger);
  }
}

TEST(FifoTest, HostProxyFifo) {
  mscclpp::HostProxyFifo hostFifo;

  kernelFifoTestTrigger<<<1, 1>>>(hostFifo.deviceFifo());

  mscclpp::ProxyTrigger trigger;
  trigger.fst = 0;
  trigger.snd = 0;

  uint64_t spin = 0;
  double start = now();
  for (uint64_t i = 0; i < ITER; ++i) {
    while (trigger.fst == 0) {
      hostFifo.poll(&trigger);

      if (spin++ > 100000) {
        FAIL() << "Program is stuck.";
      }
    }
    ASSERT_TRUE(trigger.fst != (i + 1) || trigger.snd != (i + 1));
    hostFifo.pop();
    trigger.fst = 0;
    spin = 0;
  }
}
