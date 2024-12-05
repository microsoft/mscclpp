// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_DATA_TYPES_HPP_
#define MSCCLPP_GPU_DATA_TYPES_HPP_

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

using __bfloat16 = __hip_bfloat16;
using __bfloat162 = __hip_bfloat162;
#define __CUDA_BF16_TYPES_EXIST__

#else

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#if (CUDART_VERSION >= 11000)
#include <cuda_bf16.h>
#endif
#if (CUDART_VERSION >= 11080)
#include <cuda_fp8.h>
#endif

using __bfloat16 = __nv_bfloat16;
using __bfloat162 = __nv_bfloat162;

#endif

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
