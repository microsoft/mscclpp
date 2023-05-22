#include <gtest/gtest.h>
#include <mscclpp/cuda_utils.hpp>

TEST(CudaMemoryTest, Shared) {
  auto p1 = mscclpp::allocSharedCuda<uint32_t>();
  auto p2 = mscclpp::allocSharedCuda<int64_t>(5);
}

TEST(CudaMemoryTest, Unique) {
  auto p1 = mscclpp::allocUniqueCuda<uint32_t>();
  auto p2 = mscclpp::allocUniqueCuda<int64_t>(5);
}
