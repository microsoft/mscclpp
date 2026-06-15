// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_utils.hpp>

#include "../framework.hpp"

TEST(GpuUtilsTest, StreamPool) {
  auto streamPool = mscclpp::gpuStreamPool();
  cudaStream_t s;
  {
    auto stream1 = streamPool->getStream();
    s = stream1;
    EXPECT_NE(s, nullptr);
  }
  {
    auto stream2 = streamPool->getStream();
    EXPECT_EQ(cudaStream_t(stream2), s);
  }
  {
    auto stream3 = streamPool->getStream();
    auto stream4 = streamPool->getStream();
    EXPECT_NE(cudaStream_t(stream3), cudaStream_t(stream4));
  }
  streamPool->clear();
}

TEST(GpuUtilsTest, AllocShared) {
  auto p1 = mscclpp::detail::gpuCallocShared<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocShared<int64_t>(5);
}

TEST(GpuUtilsTest, AllocUnique) {
  auto p1 = mscclpp::detail::gpuCallocUnique<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocUnique<int64_t>(5);
}

TEST(GpuUtilsTest, MakeSharedHost) {
  auto p1 = mscclpp::detail::gpuCallocHostShared<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocHostShared<int64_t>(5);
}

TEST(GpuUtilsTest, MakeUniqueHost) {
  auto p1 = mscclpp::detail::gpuCallocHostUnique<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocHostUnique<int64_t>(5);
}

TEST(GpuUtilsTest, Memcpy) {
  const int nElem = 1024;
  std::vector<int> hostBuff(nElem);
  for (int i = 0; i < nElem; ++i) {
    hostBuff[i] = i + 1;
  }
  std::vector<int> hostBuffTmp(nElem, 0);
  auto devBuff = mscclpp::detail::gpuCallocShared<int>(nElem);
  mscclpp::gpuMemcpy<int>(devBuff.get(), hostBuff.data(), nElem, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<int>(hostBuffTmp.data(), devBuff.get(), nElem, cudaMemcpyDeviceToHost);

  for (int i = 0; i < nElem; ++i) {
    EXPECT_EQ(hostBuff[i], hostBuffTmp[i]);
  }
}

TEST(GpuUtilsTest, BufferPoolBasic) {
  mscclpp::GpuBufferPool pool(4096);

  auto first = pool.allocate(64, 256);
  EXPECT_EQ(first->bytes(), size_t(64));
  EXPECT_EQ(first->offset() % 256, size_t(0));
  EXPECT_EQ(first->data(), pool.data() + first->offset());
  EXPECT_EQ(first->deviceId(), pool.deviceId());
  EXPECT_EQ(pool.activeBytes(), size_t(64));

  auto second = pool.allocate(128, 512);
  EXPECT_EQ(second->bytes(), size_t(128));
  EXPECT_EQ(second->offset() % 512, size_t(0));
  EXPECT_EQ(second->data(), pool.data() + second->offset());
  EXPECT_EQ(pool.activeBytes(), size_t(64 + 128));

  first.reset();
  EXPECT_EQ(pool.activeBytes(), size_t(128));
  second.reset();
  EXPECT_EQ(pool.activeBytes(), size_t(0));
  EXPECT_EQ(pool.freeBytes(), pool.bytes());
}

TEST(GpuUtilsTest, BufferPoolReservesAlignmentPadding) {
  mscclpp::GpuBufferPool pool(1024);

  auto first = pool.allocate(100, 1);
  auto second = pool.allocate(100, 256);
  auto third = pool.allocate(1, 1);

  EXPECT_EQ(first->offset(), size_t(0));
  EXPECT_EQ(second->offset(), size_t(256));
  EXPECT_EQ(third->offset(), size_t(356));
}

TEST(GpuUtilsTest, BufferPoolReuseAfterRelease) {
  mscclpp::GpuBufferPool pool(1024);

  auto first = pool.allocate(128, 1);
  auto firstOffset = first->offset();
  first.reset();

  auto second = pool.allocate(128, 1);
  EXPECT_EQ(second->offset(), firstOffset);
  second.reset();
  EXPECT_EQ(pool.freeBytes(), pool.bytes());
}

TEST(GpuUtilsTest, BufferPoolThrowsOnInvalidAllocation) {
  mscclpp::GpuBufferPool pool(1024);

  bool zeroSizeThrows = false;
  try {
    (void)pool.allocate(0);
  } catch (const mscclpp::Error&) {
    zeroSizeThrows = true;
  }
  EXPECT_TRUE(zeroSizeThrows);

  bool zeroAlignmentThrows = false;
  try {
    (void)pool.allocate(1, 0);
  } catch (const mscclpp::Error&) {
    zeroAlignmentThrows = true;
  }
  EXPECT_TRUE(zeroAlignmentThrows);

  bool outOfMemoryThrows = false;
  try {
    (void)pool.allocate(pool.bytes() + 1);
  } catch (const mscclpp::Error&) {
    outOfMemoryThrows = true;
  }
  EXPECT_TRUE(outOfMemoryThrows);
}
