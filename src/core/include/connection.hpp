// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <atomic>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "communicator.hpp"
#include "context.hpp"
#include "endpoint.hpp"
#include "gdr.hpp"
#include "ib.hpp"
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

  static const Endpoint::Impl& getImpl(const Endpoint& endpoint);
  static const RegisteredMemory::Impl& getImpl(const RegisteredMemory& memory);
  static Context::Impl& getImpl(Context& context);

  std::shared_ptr<Context> context_;
  Endpoint localEndpoint_;
  int maxWriteQueueSize_;
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

  void flush(int64_t timeoutUsec) override;
};

class IBConnection : public BaseConnection {
 private:
  Transport transport_;
  Transport remoteTransport_;
  std::weak_ptr<IbQp> qp_;
  std::unique_ptr<uint64_t> atomicSrc_;
  RegisteredMemory atomicSrcMem_;
  mscclpp::TransportInfo atomicSrcTransportInfo_;

  // For write-with-imm mode (HostNoAtomic): uses RDMA write-with-imm to signal
  // instead of atomic operations, with a host thread forwarding to GPU for memory consistency.
  bool ibNoAtomic_;
  bool gdrSignalForwarding_;  // ibNoAtomic_ && gdrEnabled() — decided once at construction
  std::thread recvThread_;
  std::atomic<bool> stopRecvThread_;
  int localGpuDeviceId_;  // Local GPU device ID for CUDA context and GDR mapping

  // Signal forwarding design (HostNoAtomic mode):
  // - Sender: 0-byte RDMA WRITE_WITH_IMM carrying the lower 32 bits of the token in imm_data.
  // - Receiver: CPU recv thread polls recv CQ for WRITE_WITH_IMM completions (CQE), reads
  //   the lower 32 bits from imm_data, reconstructs the full 64-bit token using wrap-around
  //   detection (monotonically increasing tokens: if lower 32 bits decrease, the upper half
  //   incremented), then writes it to signalAddr_ via atomicStore through GDRCopy BAR1.
  uint64_t signalAddr_;


  std::unique_ptr<GdrMap> signalGdrMap_;

  void recvThreadFunc();

 public:
  IBConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);
  ~IBConnection();

  /// Start signal forwarding to the given memory address.
  /// Must be called before the remote sends any updateAndSync in HostNoAtomic mode.
  /// @param mem Shared pointer to the GPU memory for the signal token.
  void startSignalForwarding(std::shared_ptr<uint64_t> mem) override;

  /// Stop signal forwarding and release associated resources.
  void stopSignalForwarding() override;

  bool isSignalForwarding() const override;

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class EthernetConnection : public BaseConnection {
 private:
  std::unique_ptr<Socket> sendSocket_;
  std::unique_ptr<Socket> recvSocket_;
  std::thread threadRecvMessages_;
  volatile uint32_t* abortFlag_;
  const uint64_t sendBufferSize_;
  const uint64_t recvBufferSize_;
  std::vector<char> sendBuffer_;
  std::vector<char> recvBuffer_;

  void recvMessages();
  void sendMessage();

 public:
  EthernetConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint,
                     uint64_t sendBufferSize = 256 * 1024 * 1024, uint64_t recvBufferSize = 256 * 1024 * 1024);

  ~EthernetConnection();

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
