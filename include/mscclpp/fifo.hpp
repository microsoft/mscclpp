// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

#include <memory>

#include "fifo_device.hpp"

namespace mscclpp {

constexpr size_t DEFAULT_FIFO_SIZE = 256;

/// Host-side proxy FIFO for device-produced work elements.
class Fifo {
 public:
  /// Construct a FIFO with a given number of entries.
  /// @param size Number of entries (default: 256).
  Fifo(int size = DEFAULT_FIFO_SIZE);

  /// Destructor.
  ~Fifo();

  /// Polls the FIFO for a trigger.
  ///
  /// @return A ProxyTrigger which is the trigger at the head of fifo.
  ProxyTrigger poll();

  /// Remove the head trigger.
  void pop();

  /// Get FIFO size.
  /// @return Number of entries in the FIFO.
  int size() const;

  /// Get device-side FIFO handle.
  /// @return FifoDeviceHandle for device access.
  FifoDeviceHandle deviceHandle() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
