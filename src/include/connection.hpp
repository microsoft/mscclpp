#ifndef MSCCLPP_CONNECTION_HPP_
#define MSCCLPP_CONNECTION_HPP_

// TODO(saemal): make this configurable
#define MSCCLPP_POLLING_WAIT 3e7  // in microseconds

#include <cuda_runtime.h>

#include <mscclpp/core.hpp>

#include "communicator.hpp"
#include "ib.hpp"

namespace mscclpp {

// TODO: Add functionality to these classes for Communicator to do connectionSetup

class ConnectionBase : public Connection, public Setuppable {
  int remoteRank_;
  int tag_;

 public:
  ConnectionBase(int remoteRank, int tag);

  int remoteRank() override;
  int tag() override;
};

class CudaIpcConnection : public ConnectionBase {
  cudaStream_t stream_;

 public:
  CudaIpcConnection(int remoteRank, int tag, cudaStream_t stream);

  ~CudaIpcConnection();

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;

  void flush() override;
};

class IBConnection : public ConnectionBase {
  Transport transport_;
  Transport remoteTransport_;
  IbQp* qp;
  int numSignaledSends;

 public:
  IBConnection(int remoteRank, int tag, Transport transport, Communicator::Impl& commImpl);

  Transport transport() override;

  Transport remoteTransport() override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;

  void flush() override;

  void beginSetup(std::shared_ptr<BaseBootstrap> bootstrap) override;

  void endSetup(std::shared_ptr<BaseBootstrap> bootstrap) override;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONNECTION_HPP_
