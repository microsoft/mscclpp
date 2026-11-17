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

#include <cstdint>

namespace mscclpp {

/// Resource sharing mode for a PortChannel's device-initiated queue pair.
enum class PortChannelSharingMode : uint8_t {
  /// The queue pair is accessed by a single thread at a time (serialized by the caller).
  Exclusive = 0,
  /// The queue pair is shared among threads within a CTA.
  Cta = 1,
  /// The queue pair is shared among threads across the GPU.
  Gpu = 2,
};

}  // namespace mscclpp

#endif  // MSCCLPP_DEVICE_HPP_
