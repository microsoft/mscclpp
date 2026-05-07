// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <unordered_map>

#include "api.h"
#include "connection.hpp"
#include "debug.h"

namespace mscclpp {

// Lightweight diagnostic counters kept in a side-table keyed by ProxyService*
// so the ProxyService class layout stays ABI-compatible with prebuilt
// extensions (e.g. mscclpp/_mscclpp.cpython-*.so) that were compiled against
// the older header. Populated only when MSCCLPP_PROXY_STATS=1.
namespace {
struct ProxyStats {
  bool enabled = false;
  bool printed = false;
  uint64_t triggers = 0;
  uint64_t trigData = 0;
  uint64_t trigFlag = 0;
  uint64_t trigAtomic = 0;
  uint64_t trigSync = 0;
  uint64_t postCalls = 0;
  uint64_t idleDrains = 0;
};
// Process-wide flag: 0 unless any ProxyService has stats enabled. Avoids
// taking the side-table mutex on the proxy hot path when nobody asked for
// stats.
std::atomic<bool>& statsAnyEnabled() {
  static std::atomic<bool> v{false};
  return v;
}
std::mutex& statsMu() { static std::mutex m; return m; }
std::unordered_map<const ProxyService*, ProxyStats>& statsTable() {
  static std::unordered_map<const ProxyService*, ProxyStats> t;
  return t;
}
ProxyStats* getStats(const ProxyService* self) {
  if (!statsAnyEnabled().load(std::memory_order_relaxed)) return nullptr;
  std::lock_guard<std::mutex> lk(statsMu());
  auto it = statsTable().find(self);
  if (it == statsTable().end()) return nullptr;
  return &it->second;
}
void printAndEraseStats(const ProxyService* self) {
  std::lock_guard<std::mutex> lk(statsMu());
  auto it = statsTable().find(self);
  if (it == statsTable().end()) return;
  ProxyStats& s = it->second;
  if (s.enabled && !s.printed) {
    s.printed = true;
    uint64_t posts = s.postCalls + s.idleDrains;
    double wrPerPost = posts ? static_cast<double>(s.triggers) / static_cast<double>(posts) : 0.0;
    fprintf(stderr,
            "[mscclpp proxy stats] triggers=%llu (data=%llu flag=%llu atomic=%llu sync=%llu) "
            "postCalls=%llu idleDrains=%llu triggersPerPost=%.2f\n",
            (unsigned long long)s.triggers, (unsigned long long)s.trigData,
            (unsigned long long)s.trigFlag, (unsigned long long)s.trigAtomic,
            (unsigned long long)s.trigSync, (unsigned long long)s.postCalls,
            (unsigned long long)s.idleDrains, wrPerPost);
    fflush(stderr);
  }
  statsTable().erase(it);
}
}  // namespace

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId,
                                                 std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(proxy) {}

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
  if (const char* env = std::getenv("MSCCLPP_PROXY_STATS")) {
    if (std::atoi(env) > 0) {
      {
        std::lock_guard<std::mutex> lk(statsMu());
        statsTable()[this].enabled = true;
      }
      statsAnyEnabled().store(true, std::memory_order_relaxed);
    }
  }
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
      INFO(MSCCLPP_INIT, "NUMA node of ProxyService proxy thread is set to %d", deviceNumaNode);
    }
  };
  auto handlerFunc = [&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); };
  proxy_ = std::make_shared<Proxy>(handlerFunc, initFunc, fifoSize);
  // Drain any deferred ibv_post_send work the moment the FIFO empties out, so
  // we batch as many triggers as possible into a single syscall while still
  // keeping latency within one FIFO drain.
  proxy_->setOnIdle([this]() { this->postPendingAll(); });
}

void ProxyService::postPendingAll() {
  if (!stagedConns_.empty()) {
    if (auto* s = getStats(this); s && s->enabled) s->idleDrains++;
  }
  for (auto& kv : stagedConns_) {
    kv.first->postPending();
  }
  stagedConns_.clear();
  dirtyConns_.clear();
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
  printAndEraseStats(this);
}

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger trigger) {
  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger.fields.semaphoreId];

  ProxyStats* stats = getStats(this);
  if (stats && stats->enabled) {
    stats->triggers++;
    if (trigger.fields.type == 0) stats->trigAtomic++;
    if (trigger.fields.type & TriggerData) stats->trigData++;
    if (trigger.fields.type & TriggerFlag) stats->trigFlag++;
    if (trigger.fields.type & TriggerSync) stats->trigSync++;
  }

  auto& conn = semaphore->connection();
  int maxWriteQueueSize = conn.getMaxWriteQueueSize();
  auto connImpl = BaseConnection::getImpl(conn);
  auto& numRequests = inflightRequests_[connImpl];

  // Batch threshold: how many staged WRs we let accumulate on one connection
  // before forcing an ibv_post_send. Lower values reduce tail latency for
  // signalling triggers; higher values reduce syscalls.
  // Override via MSCCLPP_PROXY_BATCH_THRESHOLD (1 = post every trigger,
  // matching the historical behavior; the empirical sweep on H100/IB shows
  // LL traffic is wire-latency bound rather than syscall bound, so deeper
  // batching does not help and may regress dispatch tail latency).
  static const int kPostBatchThreshold = []() {
    if (const char* env = std::getenv("MSCCLPP_PROXY_BATCH_THRESHOLD")) {
      int v = std::atoi(env);
      if (v >= 1) return v;
    }
    return 1;
  }();
  bool stagedWork = false;
  // Atomic/signal triggers convey completion semantics; never defer them.
  bool isSignal = (trigger.fields.type == 0) || (trigger.fields.type & TriggerFlag);

  if (trigger.fields.type == 0) {
    // type == 0 indicates an atomic add operation.
    // The full 64-bit add value is encoded in fst (size + srcOffset fields).
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    int64_t value = static_cast<int64_t>(trigger.fst);
    conn.atomicAdd(dst, trigger.fields.dstOffset, value);
    numRequests++;
    stagedWork = true;
  }

  if (trigger.fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger.fields.srcMemoryId];
    conn.write(dst, trigger.fields.dstOffset, src, trigger.fields.srcOffset, trigger.fields.size);
    numRequests++;
    stagedWork = true;
  }

  if (trigger.fields.type & TriggerFlag) {
    semaphore->signal();
    numRequests++;
    stagedWork = true;
  }

  if (stagedWork) {
    stagedConns_[connImpl]++;
    dirtyConns_.insert(connImpl);
  }

  bool needFlush = (trigger.fields.type & TriggerSync) && numRequests > 0;
  bool needFlush2 = maxWriteQueueSize != -1 && numRequests >= maxWriteQueueSize;
  if (needFlush || needFlush2) {
    // flush() drains staged WRs and waits for completion.
    conn.flush();
    numRequests = 0;
    stagedConns_.erase(connImpl);
    dirtyConns_.erase(connImpl);
  } else if (isSignal || stagedConns_[connImpl] >= kPostBatchThreshold) {
    // Post the staged batch now: either we just queued a completion-signal
    // WR (don't make the receiver wait for the next idle drain), or we've
    // accumulated enough WRs that the QP send queue might overflow.
    if (stats && stats->enabled) {
      stats->postCalls++;
    }
    connImpl->postPending();
    stagedConns_.erase(connImpl);
    dirtyConns_.erase(connImpl);
  }

  return ProxyHandlerResult::Continue;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  return BasePortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle());
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  return PortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(), dst_,
                                   src_);
}

}  // namespace mscclpp
