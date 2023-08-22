// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"

#include <mscclpp/utils.hpp>
#include <sstream>

#include "debug.h"
#include "endpoint.hpp"
#include "infiniband/verbs.h"
#include "npkit/npkit.h"
#include "registered_memory.hpp"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, Transport transport) {
  if (!mem.transports().has(transport)) {
    throw Error("RegisteredMemory does not support this transport", ErrorCode::InvalidUsage);
  }
}

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, cudaStream_t stream)
    : stream_(stream) {
  if (localEndpoint.transport() != Transport::CudaIpc) {
    throw mscclpp::Error("Cuda IPC connection can only be made from a Cuda IPC endpoint", ErrorCode::InvalidUsage);
  }
  if (remoteEndpoint.transport() != Transport::CudaIpc) {
    throw mscclpp::Error("Cuda IPC connection can only be made to a Cuda IPC endpoint", ErrorCode::InvalidUsage);
  }
  // sanity check: make sure the IPC connection is being made within a node
  if (remoteEndpoint.pimpl->hostHash_ != localEndpoint.pimpl->hostHash_) {
    std::stringstream ss;
    ss << "Cuda IPC connection can only be made within a node: " << std::hex << remoteEndpoint.pimpl->hostHash_
       << " != " << std::hex << localEndpoint.pimpl->hostHash_;
    throw mscclpp::Error(ss.str(), ErrorCode::InvalidUsage);
  }
  INFO(MSCCLPP_P2P, "Cuda IPC connection created");
}

Transport CudaIpcConnection::transport() { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() { return Transport::CudaIpc; }

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream_));
  INFO(MSCCLPP_P2P, "CudaIpcConnection write: from %p to %p, size %lu", srcPtr + srcOffset, dstPtr + dstOffset, size);

  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  validateTransport(dst, remoteTransport());
  uint64_t oldValue = *src;
  *src = newValue;
  uint64_t* dstPtr = (uint64_t*)dst.data();

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, src, sizeof(uint64_t), cudaMemcpyHostToDevice, stream_));
  INFO(MSCCLPP_P2P, "CudaIpcConnection atomic write: from %p to %p, %lu -> %lu", src, dstPtr + dstOffset, oldValue,
       newValue);

  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::flush(int64_t timeoutUsec) {
  if (timeoutUsec >= 0) {
    INFO(MSCCLPP_P2P, "CudaIpcConnection flush: timeout is not supported, ignored");
  }
  AvoidCudaGraphCaptureGuard guard;
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream_));
  // npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
  INFO(MSCCLPP_P2P, "CudaIpcConnection flushing connection");
}

// IBConnection

IBConnection::IBConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, Context& context)
    : transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      numSignaledSends(0),
      dummyAtomicSource_(std::make_unique<uint64_t>(0)) {
  qp = localEndpoint.pimpl->ibQp_;
  qp->rtr(remoteEndpoint.pimpl->ibQpInfo_);
  qp->rts();
  dummyAtomicSourceMem_ = context.registerMemory(dummyAtomicSource_.get(), sizeof(uint64_t), transport_);
  INFO(MSCCLPP_NET, "IB connection via %s created", getIBDeviceName(transport_).c_str());
}

Transport IBConnection::transport() { return transport_; }

Transport IBConnection::remoteTransport() { return remoteTransport_; }

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstTransportInfo = dst.pimpl->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }
  auto srcTransportInfo = src.pimpl->getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    throw Error("src is remote, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp->stageSend(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset,
                /*signaled=*/true);
  numSignaledSends++;

  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection write: from %p to %p, size %lu", (uint8_t*)srcMr->getBuff() + srcOffset,
       (uint8_t*)dstMrInfo.addr + dstOffset, size);
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)size);
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = dst.pimpl->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  qp->stageAtomicAdd(dstTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset, newValue - oldValue);
  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection atomic Write: from %p to %p, %lu -> %lu", src, (uint8_t*)dstMrInfo.addr + dstOffset,
       oldValue, newValue);
}

void IBConnection::flush(int64_t timeoutUsec) {
  Timer timer;
  while (numSignaledSends) {
    int wcNum = qp->pollCq();
    if (wcNum < 0) {
      throw mscclpp::IbError("pollCq failed: error no " + std::to_string(errno), errno);
    }

    auto elapsed = timer.elapsed();
    if ((timeoutUsec >= 0) && (elapsed * 1e3 > timeoutUsec)) {
      throw Error("pollCq is stuck: waited for " + std::to_string(elapsed / 1e3) + " seconds. Expected " +
                      std::to_string(numSignaledSends) + " signals",
                  ErrorCode::InternalError);
    }
    for (int i = 0; i < wcNum; ++i) {
      const ibv_wc* wc = qp->getWc(i);
      if (wc->status != IBV_WC_SUCCESS) {
        throw mscclpp::IbError("pollCq failed: status " + std::to_string(wc->status), wc->status);
      }
      if (wc->opcode == IBV_WC_RDMA_WRITE) {
        numSignaledSends--;
      }
    }
  }
  INFO(MSCCLPP_NET, "IBConnection flushing connection");
  // npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
}

}  // namespace mscclpp
