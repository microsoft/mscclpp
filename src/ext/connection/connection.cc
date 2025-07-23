#include "ext/connection/connection.hpp"

#include "connection.hpp"

namespace mscclpp {


void IndirectConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
  if (dstOffset + size > dst.size() || srcOffset + size > src.size()) {
    throw Error("IndirectionConnection::write out of bounds", ErrorCode::InvalidUsage);
  }
  scheduler_ptr_->sched(dst, dstOffset, src, srcOffset, size);
}

void IndirectConnection::flush(int64_t timeoutUsec) {
  if (timeoutUsec != -1) {
    throw std::runtime_error("IndirectConnection does not support timeout in flush");
  }
  scheduler_ptr_->sync();
}
Transport IndirectConnection::transport() const { return Transport::CudaIpc; }
Transport IndirectConnection::remoteTransport() const { return Transport::CudaIpc; }

}  // namespace mscclpp