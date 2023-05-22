#include <gtest/gtest.h>
#include <mscclpp/cuda_utils.hpp>

TEST(CudaMemoryTest, Shared) {
  auto p1 = mscclpp::makeSharedCuda<uint32_t>();
  auto p2 = mscclpp::makeSharedCuda<int64_t>(5);
}

TEST(CudaMemoryTest, Unique) {
  auto p1 = mscclpp::makeUniqueCuda<uint32_t>();
  auto p2 = mscclpp::makeUniqueCuda<int64_t>(5);
}
