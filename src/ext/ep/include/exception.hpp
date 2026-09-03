// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_EXCEPTION_HPP_
#define MSCCLPP_EP_EXCEPTION_HPP_

#include <exception>
#include <mscclpp/assert_device.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <string>

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
  }

  const char* what() const noexcept override { return message.c_str(); }
};

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
