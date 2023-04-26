#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include "mscclpp.hpp"
#include <cuda_runtime.h>
#include "ib.h"

namespace mscclpp {

// TODO: Add functionality to these classes for Communicator to do connectionSetup

class CudaIpcConnection : public Connection {
  cudaStream_t stream;
public:

  CudaIpcConnection();

  virtual ~CudaIpcConnection();

  virtual TransportFlags transport();

  virtual TransportFlags remoteTransport();

  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size);

  virtual void flush();
};

class IBConnection : public Connection {
  TransportFlags transport_;
  TransportFlags remoteTransport_;
  mscclppIbQp* qp;
public:

  IBConnection(TransportFlags transport, Communicator::Impl& commImpl);

  virtual ~IBConnection();

  virtual TransportFlags transport();

  virtual TransportFlags remoteTransport();

  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size);

  virtual void flush();
};

} // namespace mscclpp

#endif // MSCCLPP_CONNECTION_HPP_
