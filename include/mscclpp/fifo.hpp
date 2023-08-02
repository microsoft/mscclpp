// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <mscclpp/fifo_device.hpp>
#include <mscclpp/poll.hpp>

#define MSCCLPP_PROXY_FIFO_SIZE 128

namespace mscclpp {

/// A class representing a host proxy FIFO that can consume work elements pushed by device threads.
class Fifo {
 public:
  /// Constructs a new @ref Fifo object.
  Fifo();

  /// Destroys the @ref Fifo object.
  ~Fifo();

  /// Polls the FIFO for a trigger.
  ///
  /// @param trigger A pointer to the trigger to be filled.
  void poll(ProxyTrigger* trigger);

  /// Pops a trigger from the FIFO.
  void pop();

  /// Flushes the tail of the FIFO.
  ///
  /// @param sync If true, waits for the flush to complete before returning.
  void flushTail(bool sync = false);

  /// Returns a @ref FifoDeviceHandle object representing the device FIFO.
  ///
  /// @return A @ref FifoDeviceHandle object representing the device FIFO.
  FifoDeviceHandle deviceHandle();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
