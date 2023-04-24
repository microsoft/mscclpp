/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_CHECKS_HPP_
#define MSCCLPP_CHECKS_HPP_

#include "debug.h"
#include <cuda_runtime.h>

#define MSCCLPPTHROW(call)                                                                                             \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      throw std::runtime_error(std::string("Call to " #call " failed with error code ") + mscclppGetErrorString(res)); \
    }                                                                                                                  \
  } while (0);

#define CUDATHROW(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      throw std::runtime_error(std::string("Cuda failure '") + cudaGetErrorString(err) + "'");                         \
    }                                                                                                                  \
  } while (false)

#endif
