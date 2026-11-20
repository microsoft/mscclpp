// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <atomic>
#include <exception>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "communicator.hpp"
#include "context.hpp"
#include "endpoint.hpp"
#include "logger.hpp"
#include "registered_memory.hpp"
#include "socket.h"

namespace mscclpp {

/// Internal base class for connection implementations between two processes.
class BaseConnection {
 public:
  BaseConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint);

  virtual ~BaseConnection() = default;

  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;

  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) = 0;

  virtual void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) = 0;

  virtual void flush(int64_t timeoutUsec = -1) = 0;

  /// Start signal forwarding to the given memory address.
  /// Called by the semaphore to specify where incoming signals should be written.
  /// @param mem Shared pointer to the GPU memory for the signal token.
  virtual void startSignalForwarding(std::shared_ptr<uint64_t> /*mem*/) {}

  /// Stop signal forwarding and release associated resources.
  virtual void stopSignalForwarding() {}

  /// Whether this connection uses signal forwarding (e.g., IB host-no-atomic mode).
  /// When true, the semaphore must allocate a separate inboundToken_ for the recv thread to write to.
  /// When false, the NIC writes directly to the semaphore's registered memory (e.g., via atomics).
  virtual bool isSignalForwarding() const { return false; }

  /// Whether this connection executes PortChannel operations directly on the device rather than via the host proxy.
  virtual bool isDeviceBackend() const { return false; }

  /// Create the device backend handle for a PortChannel on this connection.
  virtual DeviceBackendHandle getDeviceBackendInfo() const { return {}; }

  /// Create the device memory handle for a RegisteredMemory on this connection.
  virtual DeviceMemoryHandle getDeviceMemoryInfo(const RegisteredMemory& /*memory*/) const { return {}; }

  /// Request a flush. Subclasses that support async flush (e.g. IBConnection) override this
  /// to be a no-op and rely on progressFlush() to drive completion via the proxy thread.
  /// The default does a blocking flush(); progressFlush() then trivially returns true.
  /// @note Only call from the proxy thread.
  virtual void requestFlush() { flush(); }

  /// Progress pending async flush operations (non-blocking CQ poll).
  /// @note Only call from the proxy thread.
  /// @return true if no flush is pending (CQ fully drained or no request).
  virtual bool progressFlush() { return true; }

  /// Get pointer to the GPU-visible flush-done position (host-pinned memory).
  /// ProxyService writes "one past the highest completed FIFO position" here when the CQ
  /// drains; GPU threads spin on it (`waitFlush`) until it surpasses their own push position.
  /// @note Pointer is valid for the lifetime of this Connection.
  uint64_t* getFlushDonePtr() const { return gpuFlushDonePos_.get(); }

  virtual Transport transport() const = 0;

  virtual Transport remoteTransport() const = 0;

  std::shared_ptr<Context> context() const;

  const Device& localDevice() const;

  int getMaxWriteQueueSize() const;

  static std::shared_ptr<BaseConnection>& getImpl(Connection& conn) { return conn.impl_; }

 protected:
  friend class Context;
  friend class CudaIpcConnection;
  friend class IBConnection;
  friend class EthernetConnection;
  friend class GpuNetIoConnection;

  static const Endpoint::Impl& getImpl(const Endpoint& endpoint);
  static const RegisteredMemory::Impl& getImpl(const RegisteredMemory& memory);
  static Context::Impl& getImpl(Context& context);

  static void validateTransport(RegisteredMemory mem, Transport transport, uint64_t offset = 0, uint64_t size = 0) {
    if (!mem.transports().has(transport)) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory does not support this transport");
    }
    if (offset + size > mem.size()) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory out of bounds");
    }
  }

  static void validateAccumulateBounds(RegisteredMemory mem, uint64_t offset) {
    constexpr uint64_t wordSize = sizeof(uint64_t);
    if (offset > mem.size() || wordSize > mem.size() - offset) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory out of bounds for 64-bit accumulate");
    }
  }

  static void validateAccumulateAlignment(uintptr_t base, uint64_t offset) {
    constexpr uintptr_t alignment = alignof(uint64_t);
    uintptr_t targetAlignment = (base % alignment + offset % alignment) % alignment;
    if (targetAlignment != 0) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "accumulate destination must be naturally 8-byte aligned");
    }
  }

  std::shared_ptr<Context> context_;
  Endpoint localEndpoint_;
  int maxWriteQueueSize_;

  // GPU-visible flush-done position (host-pinned memory). ProxyService writes one past the
  // highest FIFO position whose TriggerFlush request has fully completed on this connection
  // (CQ drained for IB, synchronous flush() returned for non-IB).
  std::shared_ptr<uint64_t> gpuFlushDonePos_;
};

class CudaIpcConnection : public BaseConnection {
 private:
  std::shared_ptr<CudaIpcStream> stream_;

 public:
  CudaIpcConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;
  void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) override;

  void flush(int64_t timeoutUsec) override;
};

class EthernetConnection : public BaseConnection {
 private:
  std::unique_ptr<Socket> sendSocket_;
  std::unique_ptr<Socket> recvSocket_;
  std::thread threadRecvMessages_;
  std::atomic<bool> stopping_{false};
  std::mutex receiverErrorMutex_;
  std::exception_ptr receiverError_;
  volatile uint32_t* abortFlag_;
  const uint64_t sendBufferSize_;
  const uint64_t recvBufferSize_;
  std::vector<char> sendBuffer_;
  std::vector<char> recvBuffer_;

  void recvMessages();
  void sendMessage();
  void publishReceiverError(std::exception_ptr error) noexcept;
  void rethrowReceiverError();
  template <typename Operation>
  void runWithReceiverErrorCheck(Operation&& operation) {
    rethrowReceiverError();
    std::forward<Operation>(operation)();
    rethrowReceiverError();
  }
  bool receiveFramePart(void* ptr, int size, bool allowBoundaryEof);

 public:
  EthernetConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint,
                     uint64_t sendBufferSize = 256 * 1024 * 1024, uint64_t recvBufferSize = 256 * 1024 * 1024);

  ~EthernetConnection() noexcept;

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;
  void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) override;

  void flush(int64_t timeoutUsec) override;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
