#include "connection.hpp"
#include "checks.hpp"
#include "registered_memory.hpp"
#include "npkit.h"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, TransportFlags transport) {
  if (mem.transports() & transport == TransportNone) {
    throw std::runtime_error("mem does not support transport");
  }
}

// CudaIpcConnection

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

  auto dstPtr = dst.impl->data;
  auto srcPtr = src.impl->data;

  CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream));
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::flush() {
  CUDATHROW(cudaStreamSynchronize(stream));
  // npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
}

// IBConnection

IBConnection::IBConnection(TransportFlags transport) : transport_(transport), remoteTransport_(TransportNone) {}

TransportFlags IBConnection::transport() {
  return transport_;
}

TransportFlags IBConnection::remoteTransport() {
  return remoteTransport_;
}

IBConnection::IBConnection(TransportFlags transport, Communicator::Impl& commImpl) : transport_(transport), remoteTransport_(TransportNone) {
  MSCCLPPTHROW(mscclppIbContextCreateQp(commImpl.getIbContext(transport), &qp));
}

IBConnection::~IBConnection() {
  // TODO: Destroy QP?
}

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstMrInfo = dst.impl->getTransportInfo<mscclppIbMrInfo>(remoteTransport());
  auto srcMr = src.impl->getTransportInfo<mscclppIbMr*>(transport());

  qp->stageSend(srcMr, &dstMrInfo, (uint32_t)size,
                        /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset, /*signaled=*/false);
  int ret = qp->postSend();
  if (ret != 0) {
    // Return value is errno.
    WARN("data postSend failed: errno %d", ret);
  }
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
      struct ibv_wc* wc = &qp->wcs[i];
      if (wc->status != IBV_WC_SUCCESS) {
        WARN("wc status %d", wc->status);
        continue;
      }
      if (wc->qp_num != qp->qp->qp_num) {
        WARN("got wc of unknown qp_num %d", wc->qp_num);
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

} // namespace mscclpp
