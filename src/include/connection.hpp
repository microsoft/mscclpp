// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <atomic>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "communicator.hpp"
#include "context.hpp"
#include "endpoint.hpp"
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

  virtual Transport transport() const = 0;

  virtual Transport remoteTransport() const = 0;

  std::shared_ptr<Context> context() const;

  const Device& localDevice() const;

  int getMaxWriteQueueSize() const;

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
  std::unique_ptr<uint64_t> dummyAtomicSource_;  // not used anywhere but IB needs a source
  RegisteredMemory dummyAtomicSourceMem_;
  mscclpp::TransportInfo dstTransportInfo_;

  // For write-with-imm signal mode
  bool useWriteImmSignal_;
  std::thread recvThread_;
  std::atomic<bool> stopRecvThread_;
  int localGpuDeviceId_;  // Local GPU device ID for setting CUDA context in recv thread

  // CPU send buffer for write-with-imm (send buffer is local to connection)
  std::unique_ptr<WriteImmData[]> writeImmSendBuf_;
  RegisteredMemory writeImmSendBufMem_;
  mscclpp::TransportInfo writeImmSendBufInfo_;
  int writeImmSendBufIdx_;  // Index to next available send buffer slot
  cudaStream_t writeImmStream_;

  // Pointers to endpoint's recv buffer (owned by Endpoint::Impl)
  WriteImmData* writeImmRecvBuf_;
  IbMrInfo remoteWriteImmRecvBufMrInfo_;  // Remote peer's recv buffer MR info (from remote endpoint)

  void recvThreadFunc();

 public:
  IBConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);
  ~IBConnection();

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
