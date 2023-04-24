#include "host_connection.hpp"
#include "communicator.hpp"
#include "comm.h"
#include "mscclpp.h"
#include "mscclppfifo.h"
#include "api.h"

namespace mscclpp {

HostConnection::Impl::Impl(Communicator* comm, mscclppConn* conn) : comm(comm), conn(conn) {
  this->hostConn = conn->hostConn;
}

HostConnection::Impl::~Impl() {
  // TODO: figure out memory ownership. Does this deallocate the mscclppHostConn? Likely not.
}

MSCCLPP_API_CPP HostConnection::~HostConnection() = default;

MSCCLPP_API_CPP HostConnection::HostConnection(std::unique_ptr<Impl> p) : pimpl(std::move(p)) {}

MSCCLPP_API_CPP int HostConnection::getId() {
  return pimpl->conn->connId;
}

MSCCLPP_API_CPP BufferHandle HostConnection::registerBuffer(void* data, uint64_t size) {
  BufferHandle result;
  static_assert(sizeof(BufferHandle) == sizeof(mscclppBufferHandle_t));
  mscclppRegisterBufferForConnection(pimpl->comm->pimpl->comm, pimpl->conn->connId, data, size, reinterpret_cast<mscclppBufferHandle_t*>(&result));
  return result;
}

MSCCLPP_API_CPP int HostConnection::numLocalBuffers() {
  return pimpl->conn->bufferRegistrations.size() - 1;
}

MSCCLPP_API_CPP BufferHandle HostConnection::getLocalBuffer(int index) {
  return index + 1;
}

MSCCLPP_API_CPP int HostConnection::numRemoteBuffers() {
  return pimpl->conn->remoteBufferRegistrations.size() - 1;
}

MSCCLPP_API_CPP BufferHandle HostConnection::getRemoteBuffer(int index) {
  return index + 1;
}

MSCCLPP_API_CPP ConnectionEpoch HostConnection::getEpoch() {
  ConnectionEpoch epoch;
  static_assert(sizeof(SignalEpochId) == sizeof(mscclppDevConnSignalEpochId));
  epoch.localSignalEpochId = reinterpret_cast<SignalEpochId*>(pimpl->conn->devConn->localSignalEpochId);
  epoch.remoteSignalEpochId = reinterpret_cast<SignalEpochId*>(pimpl->conn->devConn->remoteSignalEpochId);
  epoch.waitEpochId = pimpl->conn->devConn->waitEpochId;
  return epoch;
}


MSCCLPP_API_CPP DeviceProxyFifo HostConnection::getDeviceFifo() {
  return pimpl->comm->pimpl->proxy.fifo().toDevice();
}

MSCCLPP_API_CPP void HostConnection::put(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size) {
  pimpl->hostConn->put(dst, dstOffset, src, srcOffset, size);
}

MSCCLPP_API_CPP void HostConnection::signal() {
  pimpl->hostConn->signal();
}

MSCCLPP_API_CPP void HostConnection::flush() {
  pimpl->hostConn->flush();
}

MSCCLPP_API_CPP void HostConnection::wait() {
  pimpl->hostConn->wait();
}

} // namespace mscclpp