// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "communicator.hpp"
#include "context.hpp"
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

 public:
  IBConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);

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

class BufferResource {
 public:
  virtual ~BufferResource() = default;
  virtual RegisteredMemory next_get() = 0;
  virtual RegisteredMemory next_put() = 0;
  virtual void produce() = 0;
  virtual void consume() = 0;
};

class DoubleBuffer : public BufferResource {
  std::array<RegisteredMemory, 2> bufs_;
  int cur_{0};

 public:
  DoubleBuffer(RegisteredMemory buf1, RegisteredMemory buf2) : bufs_({buf1, buf2}) {}
  RegisteredMemory next_get() override { return bufs_[cur_]; }
  RegisteredMemory next_put() override { return bufs_[cur_ ^ 1]; }
  void produce() override { cur_ ^= 1; }
  void consume() override {}
};

class IOTask {
 public:
  void *dst, *src;
  uint64_t size;
  IOTask(void *dst_, void *src_, uint64_t size_) : dst(dst_), src(src_), size(size_) {}
};

class Scheduler {
 public:
  virtual std::vector<IOTask> produce_tasks(void *dst, void *src, uint64_t size) = 0;
  virtual void launch(const std::vector<IOTask>& tasks) = 0;
  virtual void sync() = 0;
};

class VortexScheduler : public Scheduler {
  std::shared_ptr<DoubleBuffer> buf_ptr_;
  uint64_t granularity_;
  std::array<std::shared_ptr<CudaStreamWithFlags>, 2> streams_;
  Device forwarding_device_;

 public:
  VortexScheduler(std::shared_ptr<Context> context, uint64_t granularity, Device device);
  ~VortexScheduler();
  std::vector<IOTask> produce_tasks(void *dst, void *src, uint64_t size) override;
  void launch(const std::vector<IOTask>& tasks) override;
  void sync() override;
};

class IndirectConnection : public Connection {
  std::shared_ptr<Scheduler> scheduler_ptr_;

 public:
  IndirectConnection(std::shared_ptr<Context> context,
    Endpoint localEndpoint, 
    std::shared_ptr<Scheduler> scheduler) : Connection(context, localEndpoint), scheduler_ptr_(scheduler) {}
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
    uint64_t size) override;
  void flush(int64_t timeoutUsec = -1) override;
  Transport transport() const override;
  Transport remoteTransport() const override;
  
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override {
    throw std::runtime_error("IndirectConnection does not support updateAndSync");
  }
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
