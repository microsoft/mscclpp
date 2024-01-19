// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
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

// Connection

std::shared_ptr<RegisteredMemory::Impl> Connection::getImpl(RegisteredMemory& memory) { return memory.pimpl_; }

std::shared_ptr<Endpoint::Impl> Connection::getImpl(Endpoint& memory) { return memory.pimpl_; }

std::string Connection::getTransportName() {
  return TransportNames[static_cast<int>(this->transport())] + " -> " +
         TransportNames[static_cast<int>(this->remoteTransport())];
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
  if (getImpl(remoteEndpoint)->hostHash_ != getImpl(localEndpoint)->hostHash_) {
    std::stringstream ss;
    ss << "Cuda IPC connection can only be made within a node: " << std::hex << getImpl(remoteEndpoint)->hostHash_
       << " != " << std::hex << getImpl(localEndpoint)->hostHash_;
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

// NVLS

struct NvlsConnection::Impl {
  CUmemGenericAllocationHandle mcHandle_;
  size_t bufferSize_;
  CUmulticastObjectProp mcProp_;
  size_t minMcGran_;
  size_t mcGran_;
  // These are only defined for multicast (NVLS) capability
  pid_t rootPid_;
  int mcFileDesc_;

  // use this only for the root of the NVLS
  Impl(size_t bufferSize, int numDevices) {
    minMcGran_ = 0;
    mcGran_ = 0;
    mcProp_ = {};
    mcProp_.size = bufferSize;
    mcProp_.numDevices = numDevices;
    mcProp_.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_MINIMUM));
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&mcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mcProp_.size = ((mcProp_.size + mcGran_ - 1) / mcGran_) * mcGran_;
    MSCCLPP_CUTHROW(cuMulticastCreate(&mcHandle_, &mcProp_));
    mcFileDesc_ = 0;
    MSCCLPP_CUTHROW(
        cuMemExportToShareableHandle(&mcFileDesc_, mcHandle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
    rootPid_ = getpid();

    INFO(MSCCLPP_COLL, "NVLS handle created on root");
  }

  Impl(const std::vector<char>& data) {
    auto it = data.begin();
    std::copy_n(it, sizeof(*this), reinterpret_cast<char*>(this));

    int rootPidFd = syscall(SYS_pidfd_open, rootPid_, 0);
    int mcRootFileDescFd = syscall(SYS_pidfd_getfd, rootPidFd, mcFileDesc_, 0);
    MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&mcHandle_, reinterpret_cast<void*>(mcRootFileDescFd),
                                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(rootPidFd);

    INFO(MSCCLPP_COLL, "NVLS handle was imported from root");
  }
};

NvlsConnection::NvlsConnection(size_t bufferSize, int numDevices)
    : pimpl_(std::make_shared<Impl>(bufferSize, numDevices)) {}

void NvlsConnection::addDevice() {
  int cudaDeviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDeviceId));
  MSCCLPP_CUTHROW(cuMulticastAddDevice(pimpl_->mcHandle_, cudaDeviceId));

  INFO(MSCCLPP_COLL, "NVLS connection created");
}

void NvlsConnection::addDevice(int cudaDeviceId) {
  MSCCLPP_CUTHROW(cuMulticastAddDevice(pimpl_->mcHandle_, cudaDeviceId));

  INFO(MSCCLPP_COLL, "NVLS connection created");
}

NvlsConnection::NvlsConnection(const std::vector<char>& data) : pimpl_(std::make_shared<Impl>(data)) {}

std::vector<char> NvlsConnection::serialize() {
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char*>(pimpl_.get()), sizeof(*pimpl_), std::back_inserter(result));
  return result;
}

// IBConnection

IBConnection::IBConnection(Endpoint localEndpoint, Endpoint remoteEndpoint, Context& context)
    : transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      dummyAtomicSource_(std::make_unique<uint64_t>(0)) {
  qp = getImpl(localEndpoint)->ibQp_;
  qp->rtr(getImpl(remoteEndpoint)->ibQpInfo_);
  qp->rts();
  dummyAtomicSourceMem_ = context.registerMemory(dummyAtomicSource_.get(), sizeof(uint64_t), transport_);
  validateTransport(dummyAtomicSourceMem_, transport_);
  dstTransportInfo_ = getImpl(dummyAtomicSourceMem_)->getTransportInfo(transport_);
  INFO(MSCCLPP_NET, "IB connection via %s created", getIBDeviceName(transport_).c_str());
}

Transport IBConnection::transport() { return transport_; }

Transport IBConnection::remoteTransport() { return remoteTransport_; }

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstTransportInfo = getImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }
  auto srcTransportInfo = getImpl(src)->getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    throw Error("src is remote, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp->stageSend(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset,
                /*signaled=*/true);

  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection write: from %p to %p, size %lu", (uint8_t*)srcMr->getBuff() + srcOffset,
       (uint8_t*)dstMrInfo.addr + dstOffset, size);
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)size);
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  qp->stageAtomicAdd(dstTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset, newValue - oldValue, /*signaled=*/true);

  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection atomic Write: from %p to %p, %lu -> %lu", src, (uint8_t*)dstMrInfo.addr + dstOffset,
       oldValue, newValue);
}

void IBConnection::flush(int64_t timeoutUsec) {
  Timer timer;
  while (qp->getNumCqItems()) {
    int wcNum = qp->pollCq();
    if (wcNum < 0) {
      throw mscclpp::IbError("pollCq failed: error no " + std::to_string(errno), errno);
    } else if (timeoutUsec >= 0) {
      auto elapsed = timer.elapsed();
      if (elapsed > timeoutUsec) {
        throw Error("pollCq timed out: waited for " + std::to_string(elapsed / 1e6) + " seconds. Expected " +
                        std::to_string(qp->getNumCqItems()) + " signals",
                    ErrorCode::Timeout);
      }
    }
    for (int i = 0; i < wcNum; ++i) {
      const ibv_wc* wc = qp->getWc(i);
      if (wc->status != IBV_WC_SUCCESS) {
        throw mscclpp::IbError("a work item failed: status " + std::to_string(wc->status), wc->status);
      }
    }
  }
  INFO(MSCCLPP_NET, "IBConnection flushing connection");
  // npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
}

}  // namespace mscclpp
