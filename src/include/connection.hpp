// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>

#include "communicator.hpp"
#include "context.hpp"
#include "ib.hpp"
#include "registered_memory.hpp"
#include "socket.h"

namespace mscclpp {

class CudaIpcConnection : public Connection {
  cudaStream_t stream_;

 public:
  CudaIpcConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, cudaStream_t stream);

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class IBConnection : public Connection {
  Transport transport_;
  Transport remoteTransport_;
  IbQp* qp;
  std::unique_ptr<uint64_t> dummyAtomicSource_;  // not used anywhere but IB needs a source
  RegisteredMemory dummyAtomicSourceMem_;
  mscclpp::TransportInfo dstTransportInfo_;

 public:
  IBConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, Context& context);

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;
};

class EthernetConnection : public Connection {
  std::unique_ptr<Socket> sendSocket_;
  std::unique_ptr<Socket> rcvSocket_;
  std::thread threadRcvMessages_;
  bool stopRcvMessages_;
  volatile uint32_t* abortFlag_;
  static const uint64_t sendBufferSize_ = 256000000;
  static const uint64_t rcvBufferSize_ = 256000000;
  char *sendBuffer_; 
  char *rcvBuffer_;

  public:
  EthernetConnection(Endpoint localEndpoint, Endpoint remoteEndpoint);

  ~EthernetConnection();

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;

  void flush(int64_t timeoutUsec) override;

  private:
  void rcvMessages();

  void sendMessage();
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
