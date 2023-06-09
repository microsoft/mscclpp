#include <gtest/gtest.h>

#include <mscclpp/cuda_utils.hpp>

#include "numa.hpp"

TEST(NumaTest, Basic) {
  int num;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&num));
  if (num == 0) {
    return;
  }
  for (int i = 0; i < num; i++) {
    int numaNode = mscclpp::getDeviceNumaNode(i);
    EXPECT_GE(numaNode, 0);
    mscclpp::numaBind(numaNode);
  }
}
