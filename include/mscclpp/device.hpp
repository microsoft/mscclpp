// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_HPP_

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 0)) || (defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ == 1))
/// Device code (compiled by GPU-aware compilers)

#define MSCCLPP_ON_DEVICE
#define MSCCLPP_ON_HOST_DEVICE

#if defined(__CUDA_ARCH__)
#define MSCCLPP_CUDA
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#elif defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#define MSCCLPP_HIP
#define MSCCLPP_DEVICE_INLINE __device__ inline
#endif  // defined(__CUDA_ARCH__)

#define MSCCLPP_HOST_DEVICE_INLINE MSCCLPP_DEVICE_INLINE

#elif defined(__CUDA_ARCH__) || defined(__HIP_PLATFORM_AMD__)
/// Host code but perhaps mixed with device code (compiled by GPU-aware compilers)

#define MSCCLPP_ON_HOST_DEVICE

#if defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#define MSCCLPP_CUDA_HOST
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#elif defined(__HIP_PLATFORM_AMD__) && (__HIP_PLATFORM_AMD__ == 1)
#include <hip/hip_runtime.h>
#define MSCCLPP_HIP_HOST
#define MSCCLPP_DEVICE_INLINE __device__ inline
#define MSCCLPP_HOST_DEVICE_INLINE __host__ __device__ inline
#endif

#else
/// Pure host code (compiled by GPU-unaware compilers)
#define MSCCLPP_ON_HOST
#define MSCCLPP_HOST_DEVICE_INLINE inline

#endif

#endif  // MSCCLPP_DEVICE_HPP_
