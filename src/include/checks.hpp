/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_CHECKS_HPP_
#define MSCCLPP_CHECKS_HPP_

#include "debug.h"
#include <mscclpp/errors.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define MSCCLPPTHROW(call)                                                                                             \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      throw mscclpp::Error(std::string("Call to " #call " failed with error code ") + mscclppGetErrorString(res),      \
                           ErrorCode::InvalidUsage);                                                                   \
    }                                                                                                                  \
  } while (false)

#define CUDATHROW(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      throw mscclpp::CudaError(std::string("Cuda failure '") + cudaGetErrorString(err) + "'", err);                    \
    }                                                                                                                  \
  } while (false)

#define CUTHROW(cmd)                                                                                                   \
  do {                                                                                                                 \
    CUresult err = cmd;                                                                                                \
    if (err != CUDA_SUCCESS) {                                                                                         \
      const char* errStr;                                                                                              \
      cuGetErrorString(err, &errStr);                                                                                  \
      throw mscclpp::CuError(std::string("Cu failure '") + std::string(errStr) + "'", err);                            \
    }                                                                                                                  \
  } while (false)

#endif
