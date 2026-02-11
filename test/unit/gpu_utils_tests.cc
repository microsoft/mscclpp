// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "../framework.hpp"

#include <mscclpp/gpu_utils.hpp>

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
