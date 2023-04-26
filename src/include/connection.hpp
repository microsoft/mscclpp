#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

#include "mscclpp.hpp"
#include <cuda_runtime.h>
#include "ib.h"
#include "communicator.hpp"

namespace mscclpp {

// TODO: Add functionality to these classes for Communicator to do connectionSetup

class ConnectionBase : public Connection {
public:
  virtual void startSetup(Communicator&) {};
  virtual void endSetup(Communicator&) {};
};

class CudaIpcConnection : public ConnectionBase {
  cudaStream_t stream;
public:

  CudaIpcConnection();

  ~CudaIpcConnection();

  TransportFlags transport() override;

  TransportFlags remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) override;

  void flush() override;
};

class IBConnection : public ConnectionBase {
  int remoteRank;
  int tag;
  TransportFlags transport_;
  TransportFlags remoteTransport_;
  mscclppIbQp* qp;
public:

  IBConnection(int remoteRank, int tag, TransportFlags transport, Communicator::Impl& commImpl);

  ~IBConnection();

  TransportFlags transport() override;

  TransportFlags remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) override;

  void flush() override;

  void startSetup(Communicator& comm) override;

  void endSetup(Communicator& comm) override;
};

} // namespace mscclpp

#endif // MSCCLPP_CONNECTION_HPP_
