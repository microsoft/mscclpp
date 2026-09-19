// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_EXCEPTION_HPP_
#define MSCCLPP_EP_EXCEPTION_HPP_

#include <mscclpp/assert_device.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <string>

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public mscclpp::Error {
 public:
  explicit EPException(const char* name, const char* file, const int line, const std::string& error)
      : mscclpp::Error(
            std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'",
            mscclpp::ErrorCode::InvalidUsage) {}
};

#ifndef EP_THROW
#define EP_THROW(error) throw EPException("InvalidUsage", __FILE__, __LINE__, (error))
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                     \
  do {                                                           \
    if (not(cond)) {                                             \
      throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#if defined(MSCCLPP_DEVICE_COMPILE)
#define EP_DEVICE_ASSERT(cond) MSCCLPP_ASSERT_DEVICE(cond, #cond)
#else
#define EP_DEVICE_ASSERT(cond)
#endif
#endif

#endif  // MSCCLPP_EP_EXCEPTION_HPP_
