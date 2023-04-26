#include "connection.hpp"
#include "checks.hpp"
#include "registered_memory.hpp"
#include "npkit/npkit.h"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, TransportFlags transport) {
  if ((mem.transports() & transport) == TransportNone) {
    throw std::runtime_error("mem does not support transport");
  }
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

IBConnection::IBConnection(int remoteRank, int tag, TransportFlags transport, Communicator::Impl& commImpl) : remoteRank(remoteRank), tag(tag), transport_(transport), remoteTransport_(TransportNone) {
  MSCCLPPTHROW(mscclppIbContextCreateQp(commImpl.getIbContext(transport), &qp));
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

void IBConnection::startSetup(Communicator& comm) {
  comm.bootstrap().send(&qp->info, sizeof(qp->info), remoteRank, tag);
}

void IBConnection::endSetup(Communicator& comm) {
  mscclppIbQpInfo qpInfo;
  comm.bootstrap().recv(&qpInfo, sizeof(qpInfo), remoteRank, tag);
  if (qp->rtr(&qpInfo) != 0) {
    throw std::runtime_error("Failed to transition QP to RTR");
  }
  if (qp->rts() != 0) {
    throw std::runtime_error("Failed to transition QP to RTS");
  }
}

} // namespace mscclpp
