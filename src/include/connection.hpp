#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include "mscclpp.hpp"
#include <cuda_runtime.h>
#include "ib.h"
#include "communicator.hpp"

namespace mscclpp {

// TODO: Add functionality to these classes for Communicator to do connectionSetup

class CudaIpcConnection : public Connection {
  cudaStream_t stream;
public:

  CudaIpcConnection();

  ~CudaIpcConnection();

  TransportFlags transport() override;

  TransportFlags remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) override;

  void flush() override;
};

class IBConnection : public Connection {
  TransportFlags transport_;
  TransportFlags remoteTransport_;
  mscclppIbQp* qp;
public:

  IBConnection(TransportFlags transport, Communicator::Impl& commImpl);

  ~IBConnection();

  TransportFlags transport() override;

  TransportFlags remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) override;

  void flush() override;
};

} // namespace mscclpp

#endif // MSCCLPP_CONNECTION_HPP_
