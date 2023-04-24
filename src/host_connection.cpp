#include "host_connection.hpp"

namespace mscclpp {

HostConnection::Impl::Impl() : hostConn(nullptr) {}

HostConnection::Impl::~Impl() {
  // TODO: figure out memory ownership. Does this deallocate the mscclppHostConn? Likely not.
}

void HostConnection::Impl::setup(mscclppHostConn_t *hostConn) {
  this->hostConn = hostConn;
}

BufferHandle HostConnection::registerBuffer(void* data, uint64_t size) {

}

int HostConnection::numRemoteBuffers() {

}

BufferHandle HostConnection::getRemoteBuffer(int index) {

}

DeviceConnection HostConnection::toDevice(bool startProxyThread = true) {

}

void HostConnection::put(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size) {

}

void HostConnection::put(BufferHandle dst, BufferHandle src, uint64_t offset, uint64_t size) {

}

void HostConnection::signal() {

}

void HostConnection::flush() {

}

void HostConnection::wait() {

}

void HostConnection::epochIncrement() {

}

} // namespace mscclpp