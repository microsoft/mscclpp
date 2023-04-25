#include "connection.hpp"
#include "checks.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, TransportFlags transport) {
  if (mem.transports() & transport == TransportNone) {
    throw std::runtime_error("mem does not support transport");
  }
}

TransportFlags CudaIpcConnection::transport() {
  return TransportCudaIpc;
}

TransportFlags CudaIpcConnection::remoteTransport() {
  return TransportCudaIpc;
}

CudaIpcConnection::CudaIpcConnection() {
  cudaStreamCreate(&stream);
}

CudaIpcConnection::~CudaIpcConnection() {
  cudaStreamDestroy(stream);
}

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstPtr = dst.impl->getTransportData<void*>(remoteTransport());
  auto srcPtr = src.impl->getTransportData<void*>(transport());
  CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream));
  npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)dataSize);
}

void CudaIpcConnection::flush() {
  CUDATHROW(cudaStreamSynchronize(stream));
  npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
}

IBConnection::IBConnection(TransportFlags transport) : transport_(transport), remoteTransport_(TransportNone) {}

TransportFlags IBConnection::transport() {
  return transport_;
}

TransportFlags IBConnection::remoteTransport() {
  return remoteTransport_;
}

} // namespace mscclpp
