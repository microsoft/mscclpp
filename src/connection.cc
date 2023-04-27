#include "connection.hpp"
#include "checks.hpp"
#include "registered_memory.hpp"
#include "npkit/npkit.h"
#include "infiniband/verbs.h"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, TransportFlags transport) {
  if ((mem.transports() & transport) == TransportNone) {
    throw std::runtime_error("mem does not support transport");
  }
}

// Connection

std::shared_ptr<RegisteredMemory::Impl> Connection::getRegisteredMemoryImpl(RegisteredMemory& mem) {
  return mem.pimpl;
}

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection() {
  cudaStreamCreate(&stream);
}

CudaIpcConnection::~CudaIpcConnection() {
  cudaStreamDestroy(stream);
}

TransportFlags CudaIpcConnection::transport() {
  return TransportCudaIpc;
}

TransportFlags CudaIpcConnection::remoteTransport() {
  return TransportCudaIpc;
}

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream));
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::flush() {
  CUDATHROW(cudaStreamSynchronize(stream));
  // npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
}

// IBConnection

IBConnection::IBConnection(int remoteRank, int tag, TransportFlags transport, Communicator::Impl& commImpl) : remoteRank_(remoteRank), tag_(tag), transport_(transport), remoteTransport_(TransportNone) {
  qp = commImpl.getIbContext(transport)->createQp();
}

IBConnection::~IBConnection() {
  // TODO: Destroy QP?
}

TransportFlags IBConnection::transport() {
  return transport_;
}

TransportFlags IBConnection::remoteTransport() {
  return remoteTransport_;
}

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstTransportInfo = getRegisteredMemoryImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw std::runtime_error("dst is local, which is not supported");
  }
  auto srcTransportInfo = getRegisteredMemoryImpl(src)->getTransportInfo(remoteTransport());
  if (!srcTransportInfo.ibLocal) {
    throw std::runtime_error("src is remote, which is not supported");
  }
  
  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp->stageSend(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset, /*signaled=*/false);
  qp->postSend();
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)size);
}

void IBConnection::flush() {
  bool isWaiting = true;
  while (isWaiting) {
    int wcNum = qp->pollCq();
    if (wcNum < 0) {
      WARN("pollCq failed: errno %d", errno);
      continue;
    }
    for (int i = 0; i < wcNum; ++i) {
      const struct ibv_wc* wc = reinterpret_cast<const struct ibv_wc*>(qp->getWc(i));
      if (wc->status != IBV_WC_SUCCESS) {
        WARN("wc status %d", wc->status);
        continue;
      }
      if (wc->opcode == IBV_WC_RDMA_WRITE) {
        isWaiting = false;
        break;
      }
    }
  }
  // npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
}

void IBConnection::startSetup(std::shared_ptr<BaseBootstrap> bootstrap) {
  // TODO(chhwang): temporarily disabled to compile
  bootstrap->send(&qp->getInfo(), sizeof(qp->getInfo()), remoteRank_, tag_);
}

void IBConnection::endSetup(std::shared_ptr<BaseBootstrap> bootstrap) {
  IbQpInfo qpInfo;
  // TODO(chhwang): temporarily disabled to compile
  bootstrap->recv(&qpInfo, sizeof(qpInfo), remoteRank_, tag_);
  qp->rtr(qpInfo);
  qp->rts();
}

} // namespace mscclpp
