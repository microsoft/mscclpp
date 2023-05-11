#ifndef MSCCLPP_CHECKS_HPP_
#define MSCCLPP_CHECKS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <mscclpp/errors.hpp>

#include "debug.h"

#define MSCCLPPTHROW(call)                                                                                        \
  do {                                                                                                            \
    mscclppResult_t res = call;                                                                                   \
    mscclpp::ErrorCode err = mscclpp::ErrorCode::InternalError;                                                   \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                      \
      if (res == mscclppInvalidUsage) {                                                                           \
        err = mscclpp::ErrorCode::InvalidUsage;                                                                   \
      } else if (res == mscclppSystemError) {                                                                     \
        err = mscclpp::ErrorCode::SystemError;                                                                    \
      }                                                                                                           \
      throw mscclpp::Error(std::string("Call to " #call " failed. ") + __FILE__ + ":" + std::to_string(__LINE__), \
                           err);                                                                                  \
    }                                                                                                             \
  } while (false)

#define CUDATHROW(cmd)                                                                                               \
  do {                                                                                                               \
    cudaError_t err = cmd;                                                                                           \
    if (err != cudaSuccess) {                                                                                        \
      throw mscclpp::CudaError(std::string("Call to " #cmd " failed. ") + __FILE__ + ":" + std::to_string(__LINE__), \
                               err);                                                                                 \
    }                                                                                                                \
  } while (false)

#define CUTHROW(cmd)                                                                                              \
  do {                                                                                                            \
    CUresult err = cmd;                                                                                           \
    if (err != CUDA_SUCCESS) {                                                                                    \
      throw mscclpp::CuError(std::string("Call to " #cmd " failed.") + __FILE__ + ":" + std::to_string(__LINE__), \
                             err);                                                                                \
    }                                                                                                             \
  } while (false)

#endif
