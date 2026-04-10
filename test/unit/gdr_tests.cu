// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "../framework.hpp"
#include "gdr.hpp"

// GdrStatus and gdrEnabled

class GdrStatusTest : public ::mscclpp::test::TestCase {};

TEST(GdrStatusTest, StatusIsValid) {
  // gdrStatus() should return one of the defined enum values
  auto status = mscclpp::gdrStatus();
  ASSERT_TRUE(status == mscclpp::GdrStatus::Ok || status == mscclpp::GdrStatus::NotBuilt ||
              status == mscclpp::GdrStatus::Disabled || status == mscclpp::GdrStatus::DriverMissing ||
              status == mscclpp::GdrStatus::OpenFailed);
}

TEST(GdrStatusTest, EnabledConsistentWithStatus) {
  // gdrEnabled() should be true iff gdrStatus() == Ok
  EXPECT_EQ(mscclpp::gdrEnabled(), mscclpp::gdrStatus() == mscclpp::GdrStatus::Ok);
}

// GdrMap tests — only run when GDRCopy is available

class GdrMapTest : public ::mscclpp::test::TestCase {
 protected:
  void SetUp() override {
    if (!mscclpp::gdrEnabled()) {
      SKIP_TEST() << "GDRCopy not enabled on this platform.";
    }
    MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_));
    // Try creating a GDRCopy mapping to check if pin+map works on this platform.
    try {
      auto testMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
      mscclpp::GdrMap testMap(std::static_pointer_cast<void>(testMem), deviceId_);
    } catch (const std::exception&) {
      SKIP_TEST() << "GDRCopy mapping not supported on this platform.";
    }
  }

  int deviceId_ = 0;
};

TEST(GdrMapTest, BasicMapping) {
  // Allocate GPU memory via cudaMalloc (not VMM) and create a GDRCopy mapping
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);

  ASSERT_TRUE(map.valid());
  EXPECT_NE(map.hostPtr(), nullptr);
}

TEST(GdrMapTest, CopyToAndFrom) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
  ASSERT_TRUE(map.valid());

  // Write a value to GPU via GDRCopy
  uint64_t writeVal = 0xDEADBEEFCAFE0123ULL;
  map.copyTo(&writeVal, sizeof(uint64_t));

  // Read it back via GDRCopy
  uint64_t readVal = 0;
  map.copyFrom(&readVal, sizeof(uint64_t));
  EXPECT_EQ(readVal, writeVal);

  // Also verify via cudaMemcpy
  uint64_t cudaVal = 0;
  MSCCLPP_CUDATHROW(cudaMemcpy(&cudaVal, gpuMem.get(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(cudaVal, writeVal);
}

TEST(GdrMapTest, CopyToVisibleFromGpu) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
  ASSERT_TRUE(map.valid());

  // Write via GDRCopy, verify GPU sees it via cudaMemcpy
  uint64_t val = 42;
  map.copyTo(&val, sizeof(uint64_t));

  uint64_t result = 0;
  MSCCLPP_CUDATHROW(cudaMemcpy(&result, gpuMem.get(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(result, 42);
}

TEST(GdrMapTest, MultipleWritesReadBack) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
  ASSERT_TRUE(map.valid());

  // Write multiple values sequentially and verify each
  for (uint64_t i = 1; i <= 100; ++i) {
    map.copyTo(&i, sizeof(uint64_t));
    uint64_t readback = 0;
    map.copyFrom(&readback, sizeof(uint64_t));
    EXPECT_EQ(readback, i);
    if (readback != i) break;
  }
}

TEST(GdrMapTest, HostPtrIsWritable) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
  ASSERT_TRUE(map.valid());

  // Write directly through the hostPtr (volatile store)
  volatile uint64_t* ptr = reinterpret_cast<volatile uint64_t*>(map.hostPtr());
  *ptr = 12345;

  // Read back via GDRCopy
  uint64_t readback = 0;
  map.copyFrom(&readback, sizeof(uint64_t));
  EXPECT_EQ(readback, 12345);
}

TEST(GdrMapTest, HostPtrIsReadable) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
  ASSERT_TRUE(map.valid());

  // Write via GDRCopy copyTo (same BAR1 path as the read)
  uint64_t val = 99999;
  map.copyTo(&val, sizeof(uint64_t));

  // Read through the hostPtr (volatile load via BAR1)
  volatile uint64_t* ptr = reinterpret_cast<volatile uint64_t*>(map.hostPtr());
  EXPECT_EQ(*ptr, 99999);
}

TEST(GdrMapTest, DestroyDoesNotCrash) {
  auto gpuMem = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  {
    mscclpp::GdrMap map(std::static_pointer_cast<void>(gpuMem), deviceId_);
    ASSERT_TRUE(map.valid());
    uint64_t val = 1;
    map.copyTo(&val, sizeof(uint64_t));
  }
  // After GdrMap is destroyed, gpuMem should still be valid
  uint64_t result = 0;
  MSCCLPP_CUDATHROW(cudaMemcpy(&result, gpuMem.get(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(result, 1);
}

// GPU kernel: polls signalFromCpu until it reaches expectedIter, then writes expectedIter to ackToHost.
// Repeats for maxIter iterations. The GPU uses system-scope acquire loads on signalFromCpu
// and plain stores to ackToHost (which is host-pinned memory visible to CPU).
__global__ void kernelGdrVisibilityPingPong(volatile uint64_t* signalFromCpu, volatile uint64_t* ackToHost,
                                            uint64_t maxIter) {
  for (uint64_t iter = 1; iter <= maxIter; ++iter) {
    // Poll until CPU writes the expected iteration value via GDRCopy BAR1
    while (*signalFromCpu < iter) {
    }
    // Ack back to CPU via host-pinned memory
    *ackToHost = iter;
  }
}

TEST(GdrMapTest, CpuGpuVisibilityPingPong) {
  const uint64_t maxIter = 10000;

  // signalBuf: GPU memory mapped via GDRCopy BAR1. CPU writes here, GPU polls.
  auto signalBuf = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap signalMap(std::static_pointer_cast<void>(signalBuf), deviceId_);
  ASSERT_TRUE(signalMap.valid());

  // ackBuf: host-pinned memory (gpuCallocHostShared). GPU writes here, CPU polls.
  auto ackBuf = mscclpp::detail::gpuCallocHostShared<uint64_t>(1);
  volatile uint64_t* ackPtr = reinterpret_cast<volatile uint64_t*>(ackBuf.get());
  *ackPtr = 0;

  // Launch kernel — it will poll signalBuf and write ackBuf for each iteration
  kernelGdrVisibilityPingPong<<<1, 1>>>(signalBuf.get(), ackBuf.get(), maxIter);
  MSCCLPP_CUDATHROW(cudaGetLastError());

  for (uint64_t iter = 1; iter <= maxIter; ++iter) {
    // CPU writes iteration value to GPU via GDRCopy BAR1
    uint64_t val = iter;
    signalMap.copyTo(&val, sizeof(uint64_t));

    // CPU polls host-pinned ack until GPU confirms it saw the value
    int spin = 0;
    while (*ackPtr < iter) {
      if (++spin > 100000000) {
        FAIL() << "GPU did not ack iteration " << iter << " (ack=" << *ackPtr << ")";
      }
    }
  }

  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  EXPECT_EQ(*ackPtr, maxIter);
}

// GPU kernel that polls a counter using system-scope acquire load.
// When counter >= expectedIter, writes ack.
__global__ void kernelCounterWait(uint64_t* counter, volatile uint64_t* ackToHost, uint64_t maxIter) {
  for (uint64_t iter = 1; iter <= maxIter; ++iter) {
    // System-scope acquire load — matches the atomicStore(relaxed) on the CPU side
    uint64_t got;
    do {
      got = mscclpp::atomicLoad(counter, mscclpp::memoryOrderAcquire);
    } while (got < iter);
    // Ack back
    *ackToHost = iter;
  }
}

// Test the GDRCopy counter pattern used by HostNoAtomic mode:
// - GPU memory allocated via gpuCallocShared (cudaMalloc)
// - GdrMap for BAR1 mapping
// - CPU writes via atomicStore(relaxed) through GDRCopy BAR1 mapping
// - GPU reads via atomicLoad with memory_order_acquire
TEST(GdrMapTest, AtomicStoreCounterPingPong) {
  const uint64_t maxIter = 10000;

  // Allocate GPU memory via gpuCallocShared
  auto counterBuf = mscclpp::detail::gpuCallocShared<uint64_t>(1);
  mscclpp::GdrMap counterMap(std::static_pointer_cast<void>(counterBuf), deviceId_);
  ASSERT_TRUE(counterMap.valid());

  // Ack buffer: host-pinned memory
  auto ackBuf = mscclpp::detail::gpuCallocHostShared<uint64_t>(1);
  volatile uint64_t* ackPtr = reinterpret_cast<volatile uint64_t*>(ackBuf.get());
  *ackPtr = 0;

  // Launch kernel — polls counterBuf with system-scope acquire load
  kernelCounterWait<<<1, 1>>>(counterBuf.get(), ackBuf.get(), maxIter);
  MSCCLPP_CUDATHROW(cudaGetLastError());

  for (uint64_t iter = 1; iter <= maxIter; ++iter) {
    // CPU writes counter via atomicStore (relaxed — GPU uses acquire on read)
    mscclpp::atomicStore(counterMap.hostPtr(), iter, mscclpp::memoryOrderRelaxed);

    // Wait for GPU ack
    int spin = 0;
    while (*ackPtr < iter) {
      if (++spin > 100000000) {
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        FAIL() << "GPU did not ack iteration " << iter;
      }
    }
  }

  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  EXPECT_EQ(*ackPtr, maxIter);
}
