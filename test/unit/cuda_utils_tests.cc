// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#include <mscclpp/gpu_utils.hpp>

TEST(CudaUtilsTest, AllocShared) {
  auto p1 = mscclpp::detail::gpuCallocShared<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocShared<int64_t>(5);
}

TEST(CudaUtilsTest, AllocUnique) {
  auto p1 = mscclpp::detail::gpuCallocUnique<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocUnique<int64_t>(5);
}

TEST(CudaUtilsTest, MakeSharedHost) {
  auto p1 = mscclpp::detail::gpuCallocHostShared<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocHostShared<int64_t>(5);
}

TEST(CudaUtilsTest, MakeUniqueHost) {
  auto p1 = mscclpp::detail::gpuCallocHostUnique<uint32_t>();
  auto p2 = mscclpp::detail::gpuCallocHostUnique<int64_t>(5);
}

TEST(CudaUtilsTest, Memcpy) {
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
