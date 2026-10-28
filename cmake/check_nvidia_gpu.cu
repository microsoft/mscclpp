// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda_runtime.h>

#include <cstdio>

__global__ void kernel() {}

int main() {
  int cnt;
  cudaError_t err = cudaGetDeviceCount(&cnt);
  if (err != cudaSuccess || cnt == 0) {
    return 1;
  }
  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    return 1;
  }
  std::printf("%d\n", properties.major * 10 + properties.minor);
  return 0;
}
