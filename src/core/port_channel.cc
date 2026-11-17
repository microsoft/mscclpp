// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>
#include <utility>

#include "api.h"
#include "atomic.hpp"
#include "connection.hpp"
#include "logger.hpp"
#include "proxy_impl.hpp"

namespace mscclpp {

namespace {
template <typename... Args>
void warnNoexcept(Args&&... args) noexcept {
  try {
    WARN(CONN, std::forward<Args>(args)...);
  } catch (...) {
  }
}
}  // namespace

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId,
                                                 std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(std::move(proxy)) {
  if (!semaphore) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null Host2DeviceSemaphore");
  }
  if (BaseConnection::getImpl(semaphore->connection())->isDeviceBackend()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "PortChannel device-mode connections do not use the host proxy; construct PortChannel without ProxyService");
  }
  if (!proxy_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null host proxy");
  }
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(std::shared_ptr<Device2DeviceSemaphore> semaphore)
    : semaphoreId_(0), semaphore_(std::move(semaphore)) {
  const auto& deviceSemaphore = std::get<std::shared_ptr<Device2DeviceSemaphore>>(semaphore_);
  if (!deviceSemaphore) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null Device2DeviceSemaphore");
  }
  deviceBackendHandle();
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(const Semaphore& semaphore) : semaphoreId_(0) {
  if (!semaphore.pimpl_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "PortChannel requires an initialized raw semaphore");
  }
  auto raw = semaphore;
  if (!BaseConnection::getImpl(raw.connection())->isDeviceBackend()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "PortChannel without a host proxy requires a Device-mode connection");
  }
  semaphore_ = std::make_shared<Device2DeviceSemaphore>(semaphore);
  deviceBackendHandle();
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : BasePortChannel(semaphoreId, std::make_shared<Host2DeviceSemaphore>(semaphore), std::move(proxy)) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, std::move(semaphore), std::move(proxy)), dst_(dst), src_(src) {}

MSCCLPP_API_CPP PortChannel::PortChannel(std::shared_ptr<Device2DeviceSemaphore> semaphore, RegisteredMemory dst,
                                         RegisteredMemory src)
    : BasePortChannel(std::move(semaphore)), dst_(0), src_(0), dstMemory_(std::move(dst)), srcMemory_(std::move(src)) {
  if (!dstMemory_.pimpl_ || !srcMemory_.pimpl_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "PortChannel requires initialized registered regions");
  }
}

MSCCLPP_API_CPP PortChannel::PortChannel(const Semaphore& semaphore, RegisteredMemory dst, RegisteredMemory src)
    : BasePortChannel(semaphore), dst_(0), src_(0), dstMemory_(std::move(dst)), srcMemory_(std::move(src)) {
  if (!dstMemory_.pimpl_ || !srcMemory_.pimpl_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "PortChannel requires initialized registered regions");
  }
}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, std::move(proxy)), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService(int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
      INFO(CONN, "NUMA node of ProxyService proxy thread is set to ", deviceNumaNode);
    }
  };
  auto handlerFunc = [&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); };
  proxy_ = std::make_shared<Proxy>(handlerFunc, initFunc, fifoSize);
  proxy_->pimpl_->setProgressHandler([this]() { progressFlushes(); });
}

MSCCLPP_API_CPP ProxyService::~ProxyService() noexcept {
  // Joining must succeed before any callback state can be destroyed.
  proxy_->stop();
  try {
    drainPendingOperations();
  } catch (const std::exception& e) {
    warnNoexcept("ProxyService teardown could not drain pending operations: ", e.what());
  } catch (...) {
    warnNoexcept("ProxyService teardown could not drain pending operations: unknown error");
  }
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               const Connection& connection) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(const Semaphore& semaphore) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(semaphore));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore) {
  semaphores_.push_back(semaphore);
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::nextMemoryId([[maybe_unused]] uint32_t count) const {
  if (count == 0) {
    throw Error("count must be greater than 0", ErrorCode::InvalidUsage);
  }
  MemoryId firstId = memories_.size();
  return firstId;
}

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP BasePortChannel ProxyService::basePortChannel(SemaphoreId id) {
  return BasePortChannel(id, semaphores_[id], proxy_);
}

MSCCLPP_API_CPP PortChannel ProxyService::portChannel(SemaphoreId id, MemoryId dst, MemoryId src) {
  return PortChannel(id, semaphores_[id], proxy_, dst, src);
}

MSCCLPP_API_CPP void ProxyService::startProxy(bool blocking) { proxy_->start(blocking); }

MSCCLPP_API_CPP void ProxyService::stopProxy() {
  proxy_->stop();
  drainPendingOperations();
}

void ProxyService::drainPendingOperations() {
  // Drain pending TriggerFlush operations. After a bounded loop, force-unblock any still-pending
  // GPU waiters with a sentinel write (UINT64_MAX > any FIFO position).
  for (int i = 0; i < 1000 && !pendingFlushPos_.empty(); ++i) {
    progressFlushes();
  }
  if (!pendingFlushPos_.empty()) {
    WARN(CONN, "stopProxy: ", pendingFlushPos_.size(), " connections still pending; writing sentinel");
    for (auto& [conn, pos] : pendingFlushPos_) {
      if (uint64_t* ptr = conn->getFlushDonePtr()) atomicStore(ptr, UINT64_MAX, memoryOrderRelease);
    }
    pendingFlushPos_.clear();
  }
  // Drain any remaining in-flight signaled posts that the proxy issued but never sync-flushed.
  for (auto& [conn, count] : inflightRequests_) {
    if (count <= 0) continue;
    try {
      conn->flush(/*timeoutUsec=*/5'000'000);
    } catch (const std::exception& e) {
      WARN(CONN, "stopProxy: flush failed for a connection (continuing): ", e.what());
    }
  }
  inflightRequests_.clear();
}

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger trigger) {
  // The proxy is the sole FIFO consumer and processes in strict push order, so the FIFO's
  // tail (between poll() and pop()) matches the value GPU's fifo_.push() returned for this
  // trigger — use it directly as our per-trigger sequence number.
  uint64_t pos = proxy_->fifo()->tail();

  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger.fields.semaphoreId];

  auto& conn = semaphore->connection();
  int maxWriteQueueSize = conn.getMaxWriteQueueSize();
  auto& numRequests = inflightRequests_[conn.impl_];

  auto put = [&]() {
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger.fields.srcMemoryId];
    conn.write(dst, trigger.fields.dstOffset, src, trigger.fields.srcOffset, trigger.fields.size);
    numRequests++;
  };
  auto signal = [&]() {
    semaphore->signal();
    numRequests++;
  };
  auto accumulate = [&]() {
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    // The operand is the full fst word, spanning the size and srcOffset fields.
    conn.accumulate(dst, trigger.fields.dstOffset, static_cast<int64_t>(trigger.fst));
    numRequests++;
  };

  bool flushRequested = false;
  switch (trigger.fields.type) {
    case TriggerPut:
      put();
      break;
    case TriggerSignal:
      signal();
      break;
    case TriggerFlush:
      flushRequested = true;
      break;
    case TriggerPutWithSignal:
      put();
      signal();
      break;
    case TriggerPutWithSignalAndFlush:
      put();
      signal();
      flushRequested = true;
      break;
    case TriggerAccumulate:
      accumulate();
      break;
    default:
      WARN(CONN, "unknown trigger opcode ", uint64_t(trigger.fields.type), ", ignoring the trigger");
      return ProxyHandlerResult::Continue;
  }

  if (flushRequested) {
    // Record this flush's FIFO position. The GPU caller is spinning on flushDonePos_ > pos;
    // progressFlushes() publishes pos+1 once the CQ drains. A later flush on the same connection
    // overwrites this entry, and the CQ drain completes them all at once.
    conn.impl_->requestFlush();
    pendingFlushPos_[conn.impl_] = pos;
    numRequests = 0;
  } else if (maxWriteQueueSize != -1 && numRequests >= maxWriteQueueSize) {
    conn.flush();  // flow-control flush stays blocking
    numRequests = 0;
  }

  return ProxyHandlerResult::Continue;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  if (const auto* host = std::get_if<std::shared_ptr<Host2DeviceSemaphore>>(&semaphore_)) {
    if (!*host) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel::deviceHandle() requires an initialized semaphore");
    }
    if (!proxy_) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel::deviceHandle() requires a non-null host proxy");
    }
    auto& conn = BaseConnection::getImpl((*host)->connection());
    if (conn->isDeviceBackend()) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "Device-mode connections do not use a host proxy");
    }
    return DeviceHandle(semaphoreId_, (*host)->deviceHandle(), proxy_->fifo()->deviceHandle(), conn->getFlushDonePtr());
  }
  THROW(CONN, Error, ErrorCode::InvalidUsage, "Device channels require deviceBackendHandle()");
}

MSCCLPP_API_CPP BasePortChannelDeviceHandleT<PortChannelBackend::Device> BasePortChannel::deviceBackendHandle() const {
  const auto* owner = std::get_if<std::shared_ptr<Device2DeviceSemaphore>>(&semaphore_);
  if (!owner || !*owner) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "deviceBackendHandle() requires an initialized Device semaphore");
  }
  const auto& device = *owner;
  if (!device->semaphore_.pimpl_ || !device->expectedInboundToken_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "BasePortChannel::deviceBackendHandle() requires an initialized semaphore");
  }
  const auto receive = device->portReceiveHandle();
  auto& conn = BaseConnection::getImpl(device->connection());
  const auto remoteSignal = conn->getDeviceMemoryInfo(device->semaphore_.remoteMemory());
  if (remoteSignal.addr == 0 || remoteSignal.addr % sizeof(uint64_t) != 0) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "remote semaphore token is not addressable in Device mode");
  }
  return BasePortChannelDeviceHandleT<PortChannelBackend::Device>(receive, conn->getDeviceBackendInfo(), remoteSignal);
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  return DeviceHandle(BasePortChannel::deviceHandle(), DeviceHandle::HostBound{dst_, src_});
}

MSCCLPP_API_CPP PortChannelDeviceHandleT<PortChannelBackend::Device> PortChannel::deviceBackendHandle() const {
  const auto base = BasePortChannel::deviceBackendHandle();
  if (!dstMemory_.pimpl_ || !srcMemory_.pimpl_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "PortChannel requires initialized registered regions");
  }
  auto& conn = BaseConnection::getImpl(std::get<std::shared_ptr<Device2DeviceSemaphore>>(semaphore_)->connection());
  using Handle = PortChannelDeviceHandleT<PortChannelBackend::Device>;
  return Handle(base,
                Handle::DeviceBound{conn->getDeviceMemoryInfo(dstMemory_), conn->getDeviceMemoryInfo(srcMemory_)});
}

void ProxyService::progressFlushes() {
  for (auto it = pendingFlushPos_.begin(); it != pendingFlushPos_.end();) {
    if (it->first->progressFlush()) {
      // CQ drained: publish pos+1 to unblock GPU waiters whose own pos <= recorded pos.
      atomicStore(it->first->getFlushDonePtr(), it->second + 1, memoryOrderRelease);
      it = pendingFlushPos_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mscclpp
