// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda_runtime.h>

#include <cstdio>

int main() {
  int cnt;
  cudaError_t err = cudaGetDeviceCount(&cnt);
  if (err != cudaSuccess || cnt == 0) {
    return 1;
  }

  for (int device = 0; device < cnt; ++device) {
    cudaDeviceProp properties;
    err = cudaGetDeviceProperties(&properties, device);
    if (err != cudaSuccess) {
      return 1;
    }
    if (device != 0) {
      std::printf(";");
    }
    std::printf("%d", properties.major * 10 + properties.minor);
  }
  return 0;
}
