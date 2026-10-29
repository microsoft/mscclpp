// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>
#include <mutex>

#include "api.h"
#include "atomic.hpp"
#include "connection.hpp"
#include "gpunetio.hpp"
#include "logger.hpp"
#include "proxy_impl.hpp"

namespace mscclpp {

struct BasePortChannel::DeviceModeState {
  std::shared_ptr<GpuNetIoConnection> connection;
  /// Outbound counter shared with the semaphore and its other channels; kept alive while the channel exists.
  std::shared_ptr<uint64_t> outboundToken;
  std::mutex mutex;
  /// Device-side memory tables built so far. Older tables stay alive for device handles that still reference them.
  std::vector<std::shared_ptr<GpuNetIoMemoryDeviceHandle>> tables;
  /// Number of proxy memories covered by `deviceHandle`; memories are only ever appended to the proxy.
  size_t numMemories = 0;
  GpuNetIoDeviceHandle deviceHandle{};
  bool built = false;
};

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId,
                                                 std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(proxy) {
  if (!semaphore_) return;
  auto connection = std::dynamic_pointer_cast<GpuNetIoConnection>(BaseConnection::getImpl(semaphore_->connection()));
  if (!connection) return;
  deviceMode_ = std::make_shared<DeviceModeState>();
  deviceMode_->connection = connection;
  // The counter belongs to the semaphore: its channels share it, and a semaphore rebuilt on the same connection
  // starts from zero even if its remote token reuses the address of a destroyed one.
  if (!semaphore_->deviceOutboundToken_) {
    CudaDeviceGuard deviceGuard(connection->localDevice().id);
    semaphore_->deviceOutboundToken_ = detail::gpuCallocShared<uint64_t>();
  }
  deviceMode_->outboundToken = semaphore_->deviceOutboundToken_;
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : BasePortChannel(semaphoreId, std::make_shared<Host2DeviceSemaphore>(semaphore), proxy) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

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
  auto& memories = proxy_->pimpl_->memories;
  memories.push_back(memory);
  return memories.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::nextMemoryId([[maybe_unused]] uint32_t count) const {
  if (count == 0) {
    throw Error("count must be greater than 0", ErrorCode::InvalidUsage);
  }
  MemoryId firstId = proxy_->pimpl_->memories.size();
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
  auto& memories = proxy_->pimpl_->memories;

  auto put = [&]() {
    RegisteredMemory& dst = memories[trigger.fields.dstMemoryId];
    RegisteredMemory& src = memories[trigger.fields.srcMemoryId];
    conn.write(dst, trigger.fields.dstOffset, src, trigger.fields.srcOffset, trigger.fields.size);
    numRequests++;
  };
  auto signal = [&]() {
    semaphore->signal();
    numRequests++;
  };
  auto accumulate = [&]() {
    RegisteredMemory& dst = memories[trigger.fields.dstMemoryId];
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

void BasePortChannel::setDeviceModeHandle(BasePortChannelDeviceHandle& handle) const {
  if (!deviceMode_) return;
  auto& state = *deviceMode_;
  const auto& memories = proxy_->pimpl_->memories;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.built || state.numMemories != memories.size()) {
    auto& connection = *state.connection;
    std::vector<GpuNetIoMemoryDeviceHandle> table;
    GpuNetIoDeviceHandle deviceHandle =
        connection.deviceHandle(semaphore_->semaphore_.remoteMemory(), state.outboundToken.get(), table, memories);
    std::shared_ptr<GpuNetIoMemoryDeviceHandle> deviceTable;
    if (!table.empty()) {
      CudaDeviceGuard deviceGuard(connection.localDevice().id);
      deviceTable = detail::gpuCallocShared<GpuNetIoMemoryDeviceHandle>(table.size());
      gpuMemcpy<GpuNetIoMemoryDeviceHandle>(deviceTable.get(), table.data(), table.size(), cudaMemcpyHostToDevice);
      state.tables.push_back(deviceTable);
    }
    deviceHandle.memories = deviceTable.get();
    state.deviceHandle = deviceHandle;
    state.numMemories = memories.size();
    state.built = true;
  }
  handle.backend_ = PortChannelBackend::Device;
  handle.device_ = state.deviceHandle;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  BasePortChannel::DeviceHandle handle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(),
                                       conn.impl_->getFlushDonePtr());
  setDeviceModeHandle(handle);
  return handle;
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  PortChannel::DeviceHandle handle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(), dst_, src_,
                                   conn.impl_->getFlushDonePtr());
  setDeviceModeHandle(handle);
  return handle;
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
