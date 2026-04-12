// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"

#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>
#include <thread>

#include "api.h"
#include "context.hpp"
#include "endpoint.hpp"
#include "gpu_utils_internal.hpp"
#include "logger.hpp"

namespace mscclpp {

static void validateTransport(RegisteredMemory mem, Transport transport, uint64_t offset = 0, uint64_t size = 0) {
  if (!mem.transports().has(transport)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory does not support this transport");
  }
  if (offset + size > mem.size()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory out of bounds");
  }
}

static bool isSameProcess(const Endpoint& a, const Endpoint& b) {
  return a.hostHash() == b.hostHash() && a.pidHash() == b.pidHash();
}

// BaseConnection

const Endpoint::Impl& BaseConnection::getImpl(const Endpoint& endpoint) { return *(endpoint.pimpl_); }

const RegisteredMemory::Impl& BaseConnection::getImpl(const RegisteredMemory& memory) { return *(memory.pimpl_); }

Context::Impl& BaseConnection::getImpl(Context& context) { return *(context.pimpl_); }

MSCCLPP_API_CPP BaseConnection::BaseConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint)
    : context_(context), localEndpoint_(localEndpoint), maxWriteQueueSize_(localEndpoint.maxWriteQueueSize()) {}

MSCCLPP_API_CPP std::shared_ptr<Context> BaseConnection::context() const { return context_; }

MSCCLPP_API_CPP const Device& BaseConnection::localDevice() const { return localEndpoint_.device(); }

MSCCLPP_API_CPP int BaseConnection::getMaxWriteQueueSize() const { return maxWriteQueueSize_; }

// Connection wrapper

Connection::Connection(std::shared_ptr<BaseConnection> impl) : impl_(impl) {}

MSCCLPP_API_CPP void Connection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                                       uint64_t srcOffset, uint64_t size) {
  impl_->write(dst, dstOffset, src, srcOffset, size);
}

MSCCLPP_API_CPP void Connection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src,
                                               uint64_t newValue) {
  impl_->updateAndSync(dst, dstOffset, src, newValue);
}

MSCCLPP_API_CPP void Connection::atomicAdd(RegisteredMemory dst, uint64_t dstOffset, int64_t value) {
  impl_->atomicAdd(dst, dstOffset, value);
}

MSCCLPP_API_CPP void Connection::flush(int64_t timeoutUsec) { impl_->flush(timeoutUsec); }

MSCCLPP_API_CPP Transport Connection::transport() const { return impl_->transport(); }

MSCCLPP_API_CPP Transport Connection::remoteTransport() const { return impl_->remoteTransport(); }

MSCCLPP_API_CPP std::shared_ptr<Context> Connection::context() const { return impl_->context(); }

MSCCLPP_API_CPP const Device& Connection::localDevice() const { return impl_->localDevice(); }

MSCCLPP_API_CPP int Connection::getMaxWriteQueueSize() const { return impl_->getMaxWriteQueueSize(); }

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                     const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint) {
  if (localEndpoint.transport() != Transport::CudaIpc || remoteEndpoint.transport() != Transport::CudaIpc) {
    THROW(CONN, Error, ErrorCode::InternalError, "CudaIpc transport is required for CudaIpcConnection");
  }
  if (localEndpoint.device().type == DeviceType::GPU && localEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError, "No GPU device ID provided for local endpoint");
  }
  if (remoteEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError, "No GPU device ID provided for remote endpoint");
  }
  int localDeviceId = localEndpoint.device().id;
  int remoteDeviceId = remoteEndpoint.device().id;
  if (localEndpoint.device().type != DeviceType::GPU && remoteEndpoint.device().type != DeviceType::GPU) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "CudaIpcConnection requires at least one GPU endpoint");
  } else if (localEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().type == DeviceType::GPU) {
    if (isSameProcess(localEndpoint, remoteEndpoint) && localDeviceId != remoteDeviceId) {
      // Connecting two GPUs in the same process - need to enable peer access explicitly
      CudaDeviceGuard deviceGuard(localDeviceId);
      auto ret = cudaDeviceEnablePeerAccess(remoteDeviceId, 0);
      if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
        MSCCLPP_CUDATHROW(ret);
      }
    }
  }
  int streamDeviceId = (localEndpoint.device().type == DeviceType::GPU) ? localDeviceId : remoteDeviceId;
  auto& ctxImpl = getImpl(*context);
#if defined(MSCCLPP_DEVICE_HIP)
  ctxImpl.ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>(streamDeviceId));
#else   // !defined(MSCCLPP_DEVICE_HIP)
  if (ctxImpl.ipcStreams_.empty()) {
    ctxImpl.ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>(streamDeviceId));
  }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
  stream_ = ctxImpl.ipcStreams_.back();
}

Transport CudaIpcConnection::transport() const { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() const { return Transport::CudaIpc; }

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  stream_->memcpyD2D(dstPtr + dstOffset, srcPtr + srcOffset, size);

  INFO(CONN, "CudaIpcConnection write: from ", srcPtr + srcOffset, " to ", dstPtr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  uint64_t oldValue = *src;
  *src = newValue;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.data()) + dstOffset);

  stream_->memcpyH2D(dstPtr + dstOffset, src, sizeof(uint64_t));

  INFO(CONN, "CudaIpcConnection atomic write: from ", src, " to ", dstPtr + dstOffset, ", ", oldValue, " -> ",
       newValue);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  if (timeoutUsec >= 0) {
    INFO(CONN, "CudaIpcConnection flush: timeout is not supported, ignored");
  }

  stream_->sync();

  INFO(CONN, "CudaIpcConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::atomicAdd(RegisteredMemory dst, uint64_t dstOffset, int64_t value) {
  validateTransport(dst, remoteTransport());
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.data()) + dstOffset);
  stream_->atomicAdd(dstPtr, value);
  INFO(CONN, "CudaIpcConnection atomicAdd: dst ", dstPtr, ", value ", value);
}

// IBConnection

void IBConnection::recvThreadFunc() {
  // Set the CUDA device context for this thread
  if (localGpuDeviceId_ >= 0) {
    cudaError_t err = cudaSetDevice(localGpuDeviceId_);
    if (err != cudaSuccess) {
      WARN(NET, "IBConnection recvThreadFunc: cudaSetDevice(", localGpuDeviceId_,
           ") failed: ", cudaGetErrorString(err));
      return;
    }
    // Bind this thread to the NUMA node of the local GPU for optimal memory access
    int deviceNumaNode = getDeviceNumaNode(localGpuDeviceId_);
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  }

  uint32_t lastImmData = 0;
  uint64_t immHighBits = 0;
  uint64_t newValueHost = 0;

  auto qp = qp_.lock();
  if (!qp) return;

  while (!stopRecvThread_.load(std::memory_order_relaxed)) {
    int wcNum = qp->pollRecvCq();
    if (wcNum < 0) {
      recvThreadErrorMsg_ = "pollRecvCq failed";
      recvThreadError_.store(true, std::memory_order_release);
      WARN(NET, "IBConnection recvThreadFunc: ", recvThreadErrorMsg_);
      break;
    }

    for (int i = 0; i < wcNum; ++i) {
      int status = qp->getRecvWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        // A failed recv WC typically means the QP entered error state (e.g., WR Flushed Error).
        // All remaining WRs will also fail — no recovery without QP recreation. Exit the thread
        // and set the error flag so the main thread can detect it.
        recvThreadErrorMsg_ = std::string("recv work completion failed: ") + qp->getRecvWcStatusString(i);
        recvThreadError_.store(true, std::memory_order_release);
        WARN(NET, "IBConnection recvThreadFunc: ", recvThreadErrorMsg_);
        return;
      }

      // Read the lower 32 bits of the token from imm_data. Reconstruct the full 64-bit value
      // using wrap-around detection: tokens increase monotonically, so if the new lower 32 bits
      // are less than the previous value, the upper 32 bits must have incremented by 1.
      uint32_t immData = qp->getRecvWcImmData(i);
      if (immData < lastImmData) {
        immHighBits += (1ULL << 32);
      }
      lastImmData = immData;
      newValueHost = immHighBits | static_cast<uint64_t>(immData);

      // Forward the token to the semaphore's inbound token address via atomicStore
      // through the GDRCopy BAR1 mapping. The GPU reads with system-scope acquire.
      if (signalAddr_ != 0) {
        if (signalGdrMap_ && signalGdrMap_->valid()) {
          atomicStore(signalGdrMap_->hostPtr(), newValueHost, memoryOrderRelaxed);
        } else {
          // For HIP/ROCm.
          // NOTE: may need a fix in the future to ensure BAR1 mapping.
          *reinterpret_cast<volatile uint64_t*>(signalAddr_) = newValueHost;
        }
      }

      // Post another recv for future messages
      qp->stageRecv(/*wrId=*/0);
      qp->postRecv();
    }
  }
}

IBConnection::IBConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                           const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint),
      transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      atomicSrc_(std::make_unique<uint64_t>(0)),
      ibNoAtomic_(getImpl(localEndpoint).ibNoAtomic_),
      gdrSignalForwarding_(false),
      stopRecvThread_(false),
      recvThreadError_(false),
      localGpuDeviceId_(localEndpoint.device().id),
      signalAddr_(0) {
  qp_ = getImpl(localEndpoint).ibQp_;
  qp_.lock()->rtr(getImpl(remoteEndpoint).ibQpInfo_);
  qp_.lock()->rts();
  atomicSrcMem_ = context->registerMemory(atomicSrc_.get(), sizeof(uint64_t), transport_);
  validateTransport(atomicSrcMem_, transport_);
  atomicSrcTransportInfo_ = getImpl(atomicSrcMem_).getTransportInfo(transport_);

  if (ibNoAtomic_) {
#if defined(MSCCLPP_USE_CUDA)
    // On CUDA, HostNoAtomic requires GDRCopy for CPU→GPU signal forwarding through BAR1.
    if (!gdrEnabled()) {
      THROW(CONN, Error, ErrorCode::InvalidUsage,
            "IB host-no-atomic mode on CUDA requires GDRCopy: ", gdrStatusMessage());
    }
    gdrSignalForwarding_ = true;
#endif  // defined(MSCCLPP_USE_CUDA)

    // On platforms with a CPU-GPU bridge that reorders posted writes (e.g., Grace/GB200
    // NVLink-C2C), HostNoAtomic requires Data Direct for correct memory ordering. Data Direct
    // routes NIC DMA through the PCIe Data Direct engine, bypassing the bridge. It is available
    // on Virtual Function (VF) devices. On platforms without such a bridge (x86, non-Grace
    // aarch64), HostNoAtomic works without Data Direct.
    //
    // We cannot reliably detect the bridge at compile time or runtime, so we emit a warning
    // when the device is not a VF. If data corruption occurs, switching to VF devices with
    // Data Direct or using IbMode::Host with RDMA atomics will resolve it.
    {
      IbCtx* ibCtx = getImpl(*context).getIbContext(transport_);
      if (!ibCtx->isVirtualFunction()) {
        WARN(CONN,
             "IB HostNoAtomic mode without a Virtual Function (VF) device may cause data corruption "
             "on platforms with a CPU-GPU bridge that reorders posted writes (e.g., Grace/GB200). "
             "Device ",
             ibCtx->getDevName(),
             " is not a VF. "
             "If you experience data corruption, use VF devices with Data Direct or IbMode::Host.");
      }
    }

    // Pre-post receive requests for incoming WRITE_WITH_IMM notifications.
    // The recv CQE guarantees the preceding data WRITE has been committed to GPU memory.
    auto qp = qp_.lock();
    int maxRecvWr = localEndpoint.config().ib.maxRecvWr;
    for (int i = 0; i < maxRecvWr; ++i) {
      qp->stageRecv(/*wrId=*/0);
    }
    qp->postRecv();
    // The recv thread is started later in startSignalForwarding() when the semaphore
    // provides the signal forwarding destination. This ensures the thread lifetime is
    // bounded by the GdrMap lifetime (created before start, destroyed after stop).
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_), " created with signal forwarding (HostNoAtomic) mode");
  } else {
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_), " created with atomic mode");
  }
}

IBConnection::~IBConnection() { stopSignalForwarding(); }

Transport IBConnection::transport() const { return transport_; }

Transport IBConnection::remoteTransport() const { return remoteTransport_; }

bool IBConnection::isSignalForwarding() const { return ibNoAtomic_; }

void IBConnection::startSignalForwarding(std::shared_ptr<uint64_t> mem) {
  // Set up the forwarding destination and GdrMap, then start the recv thread.
  // Order: set address → create GdrMap → start thread.
  signalAddr_ = reinterpret_cast<uint64_t>(mem.get());
  if (gdrSignalForwarding_) {
    signalGdrMap_ = std::make_unique<GdrMap>(std::move(mem), localGpuDeviceId_);
  }
  if (ibNoAtomic_) {
    stopRecvThread_.store(false, std::memory_order_relaxed);
    recvThread_ = std::thread([this]() { this->recvThreadFunc(); });
  }
  INFO(CONN, "IBConnection startSignalForwarding: ", (void*)signalAddr_);
}

void IBConnection::stopSignalForwarding() {
  // Stop the recv thread, then tear down GdrMap and address.
  // Order: stop thread → destroy GdrMap → clear address.
  if (ibNoAtomic_) {
    stopRecvThread_.store(true, std::memory_order_relaxed);
    if (recvThread_.joinable()) {
      recvThread_.join();
    }
  }
  if (gdrSignalForwarding_) {
    signalGdrMap_.reset();
  }
  signalAddr_ = 0;
  INFO(CONN, "IBConnection stopSignalForwarding");
}

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }
  auto srcTransportInfo = getImpl(src).getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "src is remote, which is not supported");
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp_.lock()->stageSendWrite(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset,
                             /*dstOffset=*/dstOffset, /*signaled=*/true);

  qp_.lock()->postSend();
  INFO(CONN, "IBConnection write: from ", (uint8_t*)srcMr->getBuff() + srcOffset, " to ",
       (uint8_t*)dstMrInfo.addr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  if (ibNoAtomic_) {
    // Signal forwarding: send a 0-byte RDMA WRITE_WITH_IMM with the lower 32 bits of the
    // token in imm_data. The receiver reconstructs the full 64-bit value using wrap-around
    // detection (tokens are monotonically increasing, so a decrease in the lower 32 bits
    // indicates the upper 32 bits incremented by 1).
    if (newValue <= oldValue) {
      WARN(CONN, "IBConnection signal forwarding: token is not monotonically increasing: ", oldValue, " -> ", newValue);
    } else if (newValue - oldValue >= (1ULL << 32)) {
      WARN(CONN,
           "IBConnection signal forwarding: token increment too large for 32-bit wrap-around detection: ", oldValue,
           " -> ", newValue, " (delta ", newValue - oldValue, " >= 2^32)");
    }
    unsigned int immData = static_cast<unsigned int>(newValue);
    qp_.lock()->stageSendWriteWithImm(nullptr, dstMrInfo,
                                      /*size=*/0, /*wrId=*/0,
                                      /*srcOffset=*/0, /*dstOffset=*/0,
                                      /*signaled=*/true, /*immData=*/immData);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection signal forwarding: value ", oldValue, " -> ", newValue);
  } else {
    qp_.lock()->stageSendAtomicAdd(atomicSrcTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset, newValue - oldValue,
                                   /*signaled=*/true);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection atomic write: from ", src, " to ", (uint8_t*)dstMrInfo.addr + dstOffset, ", ", oldValue,
         " -> ", newValue);
  }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Check if the recv thread has already reported an error (e.g., QP entered error state).
  if (recvThreadError_.load(std::memory_order_acquire)) {
    THROW(CONN, Error, ErrorCode::SystemError, "IBConnection recv thread failed: ", recvThreadErrorMsg_);
  }

  Timer timer;
  while (qp_.lock()->getNumSendCqItems()) {
    int wcNum = qp_.lock()->pollSendCq();
    if (wcNum < 0) {
      THROW(NET, IbError, errno, "pollSendCq failed");
    } else if (timeoutUsec >= 0) {
      auto elapsed = timer.elapsed();
      if (elapsed > timeoutUsec) {
        THROW(CONN, Error, ErrorCode::Timeout, "pollSendCq timed out: waited for ", elapsed / 1e6,
              " seconds. Expected ", qp_.lock()->getNumSendCqItems(), " signals");
      }
    }
    for (int i = 0; i < wcNum; ++i) {
      int status = qp_.lock()->getSendWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        THROW(NET, Error, ErrorCode::SystemError, "an IB work item failed: ", qp_.lock()->getSendWcStatusString(i));
      }
    }
  }
  INFO(CONN, "IBConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::atomicAdd(RegisteredMemory dst, uint64_t dstOffset, int64_t value) {
  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }
  auto dstMrInfo = dstTransportInfo.ibMrInfo;

  if (ibNoAtomic_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "atomicAdd is not supported in IB no-atomic mode");
  }

  qp_.lock()->stageSendAtomicAdd(atomicSrcTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset,
                                 static_cast<uint64_t>(value), /*signaled=*/true);
  qp_.lock()->postSend();
  INFO(CONN, "IBConnection atomicAdd: dst ", (uint8_t*)dstMrInfo.addr + dstOffset, ", value ", value);
}

// EthernetConnection

EthernetConnection::EthernetConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                       const Endpoint& remoteEndpoint, uint64_t sendBufferSize, uint64_t recvBufferSize)
    : BaseConnection(context, localEndpoint),
      abortFlag_(0),
      sendBufferSize_(sendBufferSize),
      recvBufferSize_(recvBufferSize) {
  // Validating Transport Protocol
  if (localEndpoint.transport() != Transport::Ethernet || remoteEndpoint.transport() != Transport::Ethernet) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "Ethernet connection can only be made from Ethernet endpoints");
  }

  // Instanciating Buffers
  sendBuffer_.resize(sendBufferSize_);
  recvBuffer_.resize(recvBufferSize_);

  // Creating Thread to Accept the Connection
  auto parameter = getImpl(localEndpoint).socket_.get();
  std::thread t([this, parameter]() {
    recvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
    recvSocket_->accept(parameter);
  });

  // Starting Connection
  sendSocket_ = std::make_unique<Socket>(&(getImpl(remoteEndpoint).socketAddress_), MSCCLPP_SOCKET_MAGIC,
                                         SocketTypeBootstrap, abortFlag_);
  sendSocket_->connect();

  // Ensure the Connection was Established
  t.join();

  // Starting Thread to Receive Messages
  int deviceId = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  threadRecvMessages_ = std::thread([deviceId, this]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(deviceId));
    this->recvMessages();
  });

  INFO(CONN, "Ethernet connection created");
}

EthernetConnection::~EthernetConnection() {
  sendSocket_->close();
  recvSocket_->close();
  threadRecvMessages_.join();
}

Transport EthernetConnection::transport() const { return Transport::Ethernet; }

Transport EthernetConnection::remoteTransport() const { return Transport::Ethernet; }

void EthernetConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  // Initializing Variables
  char* srcPtr = reinterpret_cast<char*>(src.data()) + srcOffset / sizeof(char);
  char* dstPtr = reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset / sizeof(char);
  uint64_t sentDataSize = 0;
  uint64_t headerSize = 0;

  // Copying Meta Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&size);
  std::copy(sizeBytes, sizeBytes + sizeof(size), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(size);

  // Getting Data From GPU and Sending Message
  while (sentDataSize < size) {
    uint64_t dataSize =
        std::min(sendBufferSize_ - headerSize / sizeof(char), (size - sentDataSize) / sizeof(char)) * sizeof(char);
    uint64_t messageSize = dataSize + headerSize;
    mscclpp::gpuMemcpy(sendBuffer_.data() + headerSize / sizeof(char), srcPtr + (sentDataSize / sizeof(char)), dataSize,
                       cudaMemcpyDeviceToHost);
    sendSocket_->send(sendBuffer_.data(), messageSize);
    sentDataSize += messageSize;
    headerSize = 0;
  }

  INFO(CONN, "EthernetConnection write: from ", srcPtr, " to ", dstPtr, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport());

  // Initializing Variables
  uint64_t oldValue = *src;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset);
  uint64_t dataSize = sizeof(uint64_t);
  uint64_t messageSize = 0;
  *src = newValue;

  // Copying Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&dataSize);
  std::copy(sizeBytes, sizeBytes + sizeof(dataSize), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dataSize);
  char* dataBytes = reinterpret_cast<char*>(src);
  std::copy(dataBytes, dataBytes + dataSize, sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += dataSize;

  // Sending Message
  sendSocket_->send(sendBuffer_.data(), messageSize);

  INFO(CONN, "EthernetConnection atomic write: from ", src, " to ", dstPtr + dstOffset, ", ", oldValue, " -> ",
       newValue);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::flush(int64_t) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  INFO(CONN, "EthernetConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::atomicAdd(RegisteredMemory dst, uint64_t dstOffset, int64_t value) {
  validateTransport(dst, remoteTransport());

  // Use the same wire format as write(): [dstPtr(8B)] [size(8B)] [data(size B)]
  // Set the MSB of size to signal atomicAdd to the receiver.
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset);
  constexpr uint64_t atomicAddFlag = uint64_t{1} << uint64_t{63};
  uint64_t dataSize = sizeof(uint64_t) | atomicAddFlag;
  uint64_t messageSize = 0;

  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + messageSize);
  messageSize += sizeof(dstPtr);

  char* sizeBytes = reinterpret_cast<char*>(&dataSize);
  std::copy(sizeBytes, sizeBytes + sizeof(dataSize), sendBuffer_.data() + messageSize);
  messageSize += sizeof(dataSize);

  char* valueBytes = reinterpret_cast<char*>(&value);
  std::copy(valueBytes, valueBytes + sizeof(value), sendBuffer_.data() + messageSize);
  messageSize += sizeof(value);

  sendSocket_->send(sendBuffer_.data(), messageSize);

  INFO(CONN, "EthernetConnection atomicAdd: dst ", dstPtr, ", value ", value);
}

void EthernetConnection::recvMessages() {
  // Declaring Variables
  char* ptr;
  uint64_t size;
  uint64_t recvSize;
  int closed = 0;
  bool received = true;
  constexpr uint64_t atomicAddFlag = uint64_t{1} << uint64_t{63};

  // Receiving Messages Until Connection is Closed
  while (recvSocket_->getState() != SocketStateClosed) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 1);
#endif

    // Receiving Data Address
    if (closed == 0) recvSocket_->recvUntilEnd(&ptr, sizeof(char*), &closed);
    received &= !closed;

    // Receiving data size (MSB may indicate atomicAdd)
    if (closed == 0) recvSocket_->recvUntilEnd(&size, sizeof(uint64_t), &closed);
    received &= !closed;

    bool isAtomicAdd = (size & atomicAddFlag) != 0;
    if (isAtomicAdd) {
      size &= ~atomicAddFlag;  // Clear flag to get actual data size
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif

    if (isAtomicAdd && received && size == sizeof(int64_t)) {
      // Atomic add: receive the value, read-modify-write on GPU memory
      int64_t addValue;
      recvSocket_->recvUntilEnd(&addValue, sizeof(int64_t), &closed);
      received &= !closed;
      if (received) {
        int64_t current;
        mscclpp::gpuMemcpy(reinterpret_cast<char*>(&current), ptr, sizeof(int64_t), cudaMemcpyDeviceToHost);
        current += addValue;
        mscclpp::gpuMemcpy(ptr, reinterpret_cast<char*>(&current), sizeof(int64_t), cudaMemcpyHostToDevice);
      }
    } else {
      // Regular write: receive data and copy to GPU
      recvSize = 0;
      while (recvSize < size && closed == 0) {
        uint64_t messageSize = std::min(recvBufferSize_, (size - recvSize) / sizeof(char)) * sizeof(char);
        recvSocket_->recvUntilEnd(recvBuffer_.data(), messageSize, &closed);
        received &= !closed;

        if (received)
          mscclpp::gpuMemcpy(ptr + (recvSize / sizeof(char)), recvBuffer_.data(), messageSize, cudaMemcpyHostToDevice);
        recvSize += messageSize;
      }
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif
  }
}

}  // namespace mscclpp
