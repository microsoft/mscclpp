/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_CHECKS_HPP_
#define MSCCLPP_CHECKS_HPP_

#include "debug.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define MSCCLPPTHROW(call)                                                                                             \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      throw std::runtime_error(std::string("Call to " #call " failed with error code ") + mscclppGetErrorString(res)); \
    }                                                                                                                  \
  } while (false)

#define CUDATHROW(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      throw std::runtime_error(std::string("Cuda failure '") + cudaGetErrorString(err) + "'");                         \
    }                                                                                                                  \
  } while (false)

#define CUTHROW(cmd)                                                                                                   \
  do {                                                                                                                 \
    CUresult err = cmd;                                                                                                \
    if (err != CUDA_SUCCESS) {                                                                                         \
      const char* errStr;                                                                                              \
      cuGetErrorString(err, &errStr);                                                                                  \
      throw std::runtime_error(std::string("Cu failure '") + std::string(errStr) + "'");                               \
    }                                                                                                                  \
  } while (false)

#endif
