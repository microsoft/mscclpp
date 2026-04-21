// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>

#include "../framework.hpp"

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
