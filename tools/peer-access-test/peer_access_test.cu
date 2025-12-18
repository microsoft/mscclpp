// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
using cudaError_t = hipError_t;
constexpr auto cudaSuccess = hipSuccess;
#define cudaGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)
#define cudaDeviceCanAccessPeer(...) hipDeviceCanAccessPeer(__VA_ARGS__)
#else
#include <cuda_runtime.h>
#endif

#include <iostream>

#define CUDACHECK(cmd)                                                \
  do {                                                                \
    cudaError_t e = cmd;                                              \
    if (e != cudaSuccess) {                                           \
      std::cerr << "Failed: " #cmd << " returned " << e << std::endl; \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)

int main() {
  bool canAccessPeerAll = true;
  int devCount = 0;
  CUDACHECK(cudaGetDeviceCount(&devCount));
  std::cout << "Detected " << devCount << " device(s)" << std::endl;
  if (devCount >= 2) {
    for (int i = 0; i < devCount; ++i) {
      for (int j = 0; j < devCount; ++j) {
        if (i != j) {
          int canAccessPeer = 0;
          CUDACHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
          if (!canAccessPeer) {
            canAccessPeerAll = false;
            std::cerr << "Device " << i << " cannot access peer Device " << j << std::endl;
          }
        }
      }
    }
  }
  if (canAccessPeerAll) {
    std::cout << "All devices can access each other" << std::endl;
  }
  return canAccessPeerAll ? 0 : 1;
}
