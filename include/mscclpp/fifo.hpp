// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

#include <memory>

#include "fifo_device.hpp"

namespace mscclpp {

constexpr size_t DEFAULT_FIFO_SIZE = 512;

/// Host-side proxy FIFO for device-produced work elements.
class Fifo {
 public:
  /// Constructor.
  /// @param size Number of entries. Must be a power of two (default: DEFAULT_FIFO_SIZE).
  /// @throws Error with ErrorCode::InvalidUsage if size is not a positive power of two.
  Fifo(int size = DEFAULT_FIFO_SIZE);

  /// Destructor.
  ~Fifo();

  /// Poll for the trigger at the head.
  ///
  /// A trigger carries no reserved payload value, so readiness is reported separately rather than
  /// encoded in the trigger itself.
  ///
  /// @param trigger Set to the trigger at the head if one is ready. Untouched otherwise.
  /// @return True if a trigger was ready and written to @p trigger.
  bool poll(ProxyTrigger& trigger);

  /// Remove the head trigger.
  void pop();

  /// Get the current tail position — the FIFO push-return value of the trigger about to be
  /// (or currently being) processed by the proxy thread. Monotonically increasing.
  /// @return The current tail position.
  uint64_t tail() const;

  /// Get FIFO size.
  /// @return Number of entries in the FIFO.
  int size() const;

  /// Get device-side FIFO handle.
  /// @return FifoDeviceHandle for device access.
  FifoDeviceHandle deviceHandle() const;

  [[deprecated("flushTail() is now no-op and no longer needed. This will be removed in a future release.")]] void
  flushTail([[maybe_unused]] bool sync = false) {}

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
