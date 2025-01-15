// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include <chrono>
#include <mscclpp/core.hpp>
#include <string>

namespace mscclpp {

struct Timer {
  std::chrono::steady_clock::time_point start_;
  int timeout_;

  Timer(int timeout = -1);

  ~Timer();

  /// Returns the elapsed time in microseconds.
  int64_t elapsed() const;

  void set(int timeout);

  void reset();

  void print(const std::string& name);
};

struct ScopedTimer : public Timer {
  const std::string name_;

  ScopedTimer(const std::string& name);

  ~ScopedTimer();
};

std::string getHostName(int maxlen, const char delim);

/// Get the number of available InfiniBand devices.
///
/// @return The number of available InfiniBand devices.
int getIBDeviceCount();

/// Get the name of the InfiniBand device associated with the specified transport.
///
/// @param ibTransport The InfiniBand transport to get the device name for.
/// @return The name of the InfiniBand device associated with the specified transport.
std::string getIBDeviceName(Transport ibTransport);

/// Get the InfiniBand transport associated with the specified device name.
///
/// @param ibDeviceName The name of the InfiniBand device to get the transport for.
/// @return The InfiniBand transport associated with the specified device name.
Transport getIBTransportByDeviceName(const std::string& ibDeviceName);

}  // namespace mscclpp

#endif  // MSCCLPP_UTILS_HPP_
