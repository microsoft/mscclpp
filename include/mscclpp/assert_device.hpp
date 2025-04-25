// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ASSERT_DEVICE_HPP_
#define MSCCLPP_ASSERT_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)

#include <cstdint>

#if !defined(DEBUG_BUILD)

#define __assert_fail(__assertion, __file, __line, __function) ;

namespace mscclpp {
MSCCLPP_DEVICE_INLINE void assert_device(bool cond, const char* msg) {}
}  // namespace mscclpp

#else  // defined(DEBUG_BUILD)

#if defined(MSCCLPP_DEVICE_HIP)
extern "C" __device__ void __assert_fail(const char *__assertion, const char *__file, unsigned int __line,
                                         const char *__function);
#else   // !defined(MSCCLPP_DEVICE_HIP)
extern "C" __host__ __device__ void __assert_fail(const char *__assertion, const char *__file, unsigned int __line,
                                                  const char *__function) __THROW;
#endif  // !defined(MSCCLPP_DEVICE_HIP)

namespace mscclpp {
MSCCLPP_DEVICE_INLINE void assert_device(bool cond, const char *msg) {
  if (!cond) {
    __assert_fail(msg, __FILE__, __LINE__, __PRETTY_FUNCTION__);
  }
}
}  // namespace mscclpp

#endif  // !defined(DEBUG_BUILD)

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_ASSERT_DEVICE_HPP_
