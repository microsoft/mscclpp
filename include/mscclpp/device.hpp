// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_HPP_

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif  // defined(__HIP_PLATFORM_AMD__)

#if (defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE __forceinline__
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#if defined(__HIP_PLATFORM_AMD__)
#define MSCCLPP_DEVICE_HIP
#else  // !(defined(__HIP_PLATFORM_AMD__))
#define MSCCLPP_DEVICE_CUDA
#endif  // defined(__HIP_PLATFORM_AMD__)

#else  // !(defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

#define MSCCLPP_HOST_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline

#endif  // !(defined(__NVCC__) || defined(__HIP_PLATFORM_AMD__))

// Whether the current device compilation target supports FP8 (`e4m3`/`e5m2`) multimem
// instructions.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
#define MSCCLPP_DEVICE_FP8_MULTIMEM_SUPPORTED 1
#else
#define MSCCLPP_DEVICE_FP8_MULTIMEM_SUPPORTED 0
#endif

#endif  // MSCCLPP_DEVICE_HPP_
