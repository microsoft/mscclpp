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
    : state_(std::in_place_type<HostState>, semaphoreId, std::move(semaphore), std::move(proxy)) {
  const auto& host = std::get<HostState>(state_);
  if (!host.semaphore) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null Host2DeviceSemaphore");
  }
  if (BaseConnection::getImpl(host.semaphore->connection())->isDeviceBackend()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "PortChannel device-mode connections do not use the host proxy; construct PortChannel without ProxyService");
  }
  if (!host.proxy) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null host proxy");
  }
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore)
    : state_(std::in_place_type<DeviceState>, std::move(semaphore)) {
  const auto& device = std::get<DeviceState>(state_);
  if (!device.semaphore) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "BasePortChannel requires a non-null PortDevice2DeviceSemaphore");
  }
  device.semaphore->deviceHandle();  // Reject a moved-from semaphore before retaining a usable-looking channel.
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : BasePortChannel(semaphoreId, std::make_shared<Host2DeviceSemaphore>(semaphore), std::move(proxy)) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, std::move(semaphore), std::move(proxy)),
      binding_(std::in_place_type<HostBinding>, dst, src) {}

MSCCLPP_API_CPP PortChannel::PortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore, RegisteredMemory dst,
                                         RegisteredMemory src)
    : BasePortChannel(std::move(semaphore)),
      binding_(std::in_place_type<DeviceBinding>, std::move(dst), std::move(src)) {
  deviceHandle<PortChannelBackend::Device>();  // Resolve and validate both registrations before publishing the channel.
}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, std::move(proxy)), binding_(std::in_place_type<HostBinding>, dst, src) {}

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

template <PortChannelBackend Backend>
MSCCLPP_API_CPP BasePortChannelDeviceHandle<Backend> BasePortChannel::deviceHandle() const
  requires(Backend == PortChannelBackend::Host || Backend == PortChannelBackend::Device)
{
  if constexpr (Backend == PortChannelBackend::Device) {
    const auto* device = std::get_if<DeviceState>(&state_);
    if (!device || !device->semaphore) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "Device extraction requires initialized Device owner state");
    }
    const auto semaphoreHandle = device->semaphore->deviceHandle();
    auto& conn = BaseConnection::getImpl(device->semaphore->connection());
    return BasePortChannelDeviceHandle<Backend>(semaphoreHandle, conn->getDeviceBackendInfo());
  } else {
    const auto* host = std::get_if<HostState>(&state_);
    if (!host || !host->semaphore || !host->proxy) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "Host extraction requires initialized Host owner state");
    }
    auto& conn = BaseConnection::getImpl(host->semaphore->connection());
    return BasePortChannelDeviceHandle<Backend>(host->id, host->semaphore->deviceHandle(),
                                                host->proxy->fifo()->deviceHandle(), conn->getFlushDonePtr());
  }
}

template <PortChannelBackend Backend>
MSCCLPP_API_CPP PortChannelDeviceHandle<Backend> PortChannel::deviceHandle() const
  requires(Backend == PortChannelBackend::Host || Backend == PortChannelBackend::Device)
{
  const auto base = BasePortChannel::deviceHandle<Backend>();
  if constexpr (Backend == PortChannelBackend::Device) {
    const auto* regions = std::get_if<DeviceBinding>(&binding_);
    if (!regions) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "Device extraction requires initialized Device bindings");
    }
    const auto& device = std::get<DeviceState>(state_);
    auto& conn = BaseConnection::getImpl(device.semaphore->connection());
    return PortChannelDeviceHandle<Backend>(base.semaphore_, base.device_, conn->getDeviceMemoryInfo(regions->dst),
                                            conn->getDeviceMemoryInfo(regions->src));
  } else {
    const auto* ids = std::get_if<HostBinding>(&binding_);
    if (!ids) {
      THROW(CONN, Error, ErrorCode::InvalidUsage, "Host extraction requires initialized Host bindings");
    }
    return PortChannelDeviceHandle<Backend>(base.semaphoreId_, base.semaphore_, base.fifo_, ids->dst, ids->src,
                                            base.flushDonePos_);
  }
}

template BasePortChannelDeviceHandle<PortChannelBackend::Host> BasePortChannel::deviceHandle<PortChannelBackend::Host>()
    const;
template BasePortChannelDeviceHandle<PortChannelBackend::Device>
BasePortChannel::deviceHandle<PortChannelBackend::Device>() const;
template PortChannelDeviceHandle<PortChannelBackend::Host> PortChannel::deviceHandle<PortChannelBackend::Host>() const;
template PortChannelDeviceHandle<PortChannelBackend::Device> PortChannel::deviceHandle<PortChannelBackend::Device>()
    const;

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
