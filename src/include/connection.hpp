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

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
