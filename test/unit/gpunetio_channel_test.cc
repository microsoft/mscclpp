#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(TEST_DEVICE_QPS)
#include <cerrno>
#define MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_USE_GPUNETIO
#define MSCCLPP_ASSERT_DEVICE_HPP_
#define MSCCLPP_ASSERT_DEVICE(condition, message) require(condition, message)
#define DOCA_GPUNETIO_DEVICE_H

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void __trap() { throw std::runtime_error("CQ error"); }
using __be32 = uint32_t;
using doca_gpu_dev_verbs_ticket_t = uint64_t;
constexpr int DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU = 0;
constexpr int DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD = 1;
uint32_t __byte_perm(uint32_t value, int, int) { return __builtin_bswap32(value); }
struct doca_gpu_dev_verbs_qp {
  uint64_t sq_rsvd_index = 7;
};
struct doca_gpu_dev_verbs_addr {
  uintptr_t addr;
  uint32_t key;
};
doca_gpu_dev_verbs_qp* selectedQp;
doca_gpu_dev_verbs_addr destination, source, signalDestination, signalResult;
uint64_t transferredBytes, signalValue, polledTicket;
int polls = 0;
int completionStatus = 0;

template <int Mode>
void doca_gpu_dev_verbs_put(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr remote, doca_gpu_dev_verbs_addr local,
                            uint64_t bytes, doca_gpu_dev_verbs_ticket_t* ticket) {
  selectedQp = qp;
  destination = remote;
  source = local;
  transferredBytes = bytes;
  *ticket = 6;
}
template <int Operation, int Mode>
void doca_gpu_dev_verbs_put_signal(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_addr remote,
                                   doca_gpu_dev_verbs_addr local, uint64_t bytes, doca_gpu_dev_verbs_addr signalRemote,
                                   doca_gpu_dev_verbs_addr signalLocal, uint64_t value,
                                   doca_gpu_dev_verbs_ticket_t* ticket) {
  doca_gpu_dev_verbs_put<Mode>(qp, remote, local, bytes, ticket);
  signalDestination = signalRemote;
  signalResult = signalLocal;
  signalValue = value;
  *reinterpret_cast<uint64_t*>(signalLocal.addr) = 99;
}
template <typename Value, int Mode>
Value doca_gpu_dev_verbs_atomic_read(Value* pointer) {
  return *pointer;
}
template <int Mode>
int doca_gpu_dev_verbs_poll_one_cq_at(doca_gpu_dev_verbs_qp* qp, uint64_t ticket) {
  selectedQp = qp;
  polledTicket = ticket;
  ++polls;
  return completionStatus;
}

#include <mscclpp/port_channel_gpunetio_device.hpp>

int main() {
  constexpr int peers = 4;
  const uint32_t keys[peers] = {11, 12, 13, 14};
  const uintptr_t bases[peers] = {1024, 2048, 3072, 4096};
  uint64_t payload[8] = {123};
  for (const int queues : {1, 2, 4, 8, 64}) {
    std::vector<doca_gpu_dev_verbs_qp> qps(peers * queues);
    std::vector<uint64_t> scratch(peers * queues, 0);
    mscclpp::GpuNetIoDeviceContext context{};
    context.qps = qps.data();
    context.rkeys = keys;
    context.peerBase = bases;
    context.localBase = reinterpret_cast<uintptr_t>(payload);
    context.lkey = 0x01020304;
    context.numPeers = peers;
    context.numQpsPerPeer = queues;
    context.atomicResultBase = reinterpret_cast<uintptr_t>(scratch.data());
    context.atomicResultLkey = 0x05060708;
    for (int peer = 0; peer < peers; ++peer) {
      for (int queue = 0; queue < queues; ++queue) {
        const int index = peer * queues + queue;
        auto* expectedQp = &qps[index];
        context.put(peer, 16, 24, 32, queue);
        require(selectedQp == expectedQp && destination.addr == bases[peer] + 16 && destination.key == keys[peer] &&
                    source.addr == context.localBase + 24 && source.key == 0x04030201 && transferredBytes == 32,
                "multi-QP put addressing");
        context.putWithSignal(peer, 16, 24, 32, 64, 5, queue);
        require(selectedQp == expectedQp && signalDestination.addr == bases[peer] + 64 &&
                    signalResult.addr == reinterpret_cast<uintptr_t>(&scratch[index]) &&
                    signalResult.key == 0x08070605 && signalValue == 5,
                "multi-QP signal scratch");
        context.atomicAdd(peer, 128, -7, queue);
        require(selectedQp == expectedQp && signalDestination.addr == bases[peer] + 128 &&
                    signalValue == static_cast<uint64_t>(-7) && transferredBytes == 0 && payload[0] == 123,
                "multi-QP atomic isolation");
        context.flush(peer, queue);
        require(selectedQp == expectedQp && polledTicket == 6, "multi-QP completion queue");
        completionStatus = EBUSY;
        polls = 0;
        require(context.tryFlush(peer, 3, queue) == EBUSY && polls == 3 && selectedQp == expectedQp,
                "bounded queue-local completion");
        completionStatus = -EIO;
        require(context.tryFlush(peer, 3, queue) == -EIO, "CQ error propagation");
        completionStatus = 0;
        require(context.tryFlush(peer, 3, queue) == 0, "CQ success");
        expectedQp->sq_rsvd_index = 0;
        polls = 0;
        context.flush(peer, queue);
        require(context.tryFlush(peer, 0, queue) == 0 && polls == 0, "empty queue completion");
      }
    }
    for (const auto result : scratch) require(result == 99, "per-QP scratch coverage");
    context.put(1, 0, 0, 8);
    require(selectedQp == &qps[queues], "legacy default QP zero");
    bool rejected = false;
    try {
      context.put(1, 0, 0, 8, queues);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "out-of-range QP accepted");
  }
  std::cout << "GPUNetIO multi-QP addressing and completion checks passed\n";
}

#elif defined(TEST_HOST_API)
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/port_channel.hpp>

static_assert(std::is_trivially_default_constructible_v<mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<mscclpp::PortChannelDeviceHandle>);

#if defined(__NVCC__)
__global__ void compileGpuNetIoChannel(mscclpp::PortChannelDeviceHandle channel) {
  channel.put(0, 8);
  channel.signal();
  channel.putWithSignal(0, 8);
  channel.putWithSignalAndFlush(0, 8, 8, 100);
  channel.accumulate(128, -1);
  channel.flush();
  channel.wait();
}
#endif

int main() {
  mscclpp::GpuNetIoDeviceContext context{};
  uint64_t inbound = 0;
  uint64_t expected = 0;
#if defined(TEST_GPUNETIO_ENABLED)
  mscclpp::PortChannel channel(&context, 3, 64, &inbound, &expected);
  auto handle = channel.deviceHandle();
  if (handle.backend_ != mscclpp::PortChannelBackend::GpuNetIo || handle.gin_ != &context || handle.ginPeer_ != 3 ||
      handle.ginSignalOffset_ != 64 || handle.semaphore_.inboundToken != &inbound ||
      handle.semaphore_.expectedInboundToken != &expected)
    return 1;
  mscclpp::BasePortChannel base(&context, 2, 128, &inbound, &expected);
  if (base.deviceHandle().ginPeer_ != 2 || base.deviceHandle().ginSignalOffset_ != 128) return 2;
  for (int invalid = 0; invalid < 6; ++invalid) {
    bool rejected = false;
    try {
      mscclpp::PortChannel bad(invalid == 0 ? nullptr : &context, invalid == 1 ? -1 : 3, invalid == 2 ? 65 : 64,
                               invalid == 3 ? nullptr : &inbound,
                               invalid == 4   ? nullptr
                               : invalid == 5 ? &inbound
                                              : &expected);
    } catch (const mscclpp::Error&) {
      rejected = true;
    }
    if (!rejected) return 3;
  }
#else
  bool rejected = false;
  try {
    mscclpp::PortChannel channel(&context, 3, 64, &inbound, &expected);
  } catch (const mscclpp::Error&) {
    rejected = true;
  }
  if (!rejected) return 4;
#endif
  std::cout << "PortChannel linked host API checks passed\n";
}

#else
#define MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_DEVICE_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline
#ifdef TEST_DEVICE_ASSERTS
#define MSCCLPP_ASSERT_DEVICE(condition, message) require(condition, message)
#else
#define MSCCLPP_ASSERT_DEVICE(condition, message)
#endif
#define POLL_MAYBE_JAILBREAK(condition, budget) require(!(condition), "unexpected polling")

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

namespace mscclpp {
enum { scopeSystem, memoryOrderAcquire };
enum { TriggerPut, TriggerSignal, TriggerPutWithSignal, TriggerPutWithSignalAndFlush, TriggerFlush, TriggerAccumulate };
constexpr int TriggerBitsSize = 32;

template <typename Value, int Scope>
Value atomicLoad(Value* pointer, int) {
  return *pointer;
}

struct ProxyTrigger {
  int type;
  uint32_t dstId;
  uint64_t dstOffset;
  uint32_t srcId;
  uint64_t srcOffset;
  uint64_t size;
  uint32_t semaphoreId;
};

struct FifoDeviceHandle {
  std::vector<ProxyTrigger>* calls = nullptr;
  uint64_t push(ProxyTrigger trigger) {
    require(calls != nullptr, "GDAKI touched proxy FIFO");
    calls->push_back(trigger);
    return calls->size() - 1;
  }
};

struct Host2DeviceSemaphoreDeviceHandle {
  uint64_t* inboundToken;
  uint64_t* expectedInboundToken;
  bool poll() {
    if (*inboundToken <= *expectedInboundToken) return false;
    ++*expectedInboundToken;
    return true;
  }
  void wait(int64_t) { require(poll(), "signal not received"); }
};

struct GpuNetIoDeviceContext {
  int numPeers = 4;
  int peer = -1;
  int puts = 0;
  int atomics = 0;
  int flushes = 0;
  int boundedFlushes = 0;
  int status = 0;
  uint64_t destination = 0;
  uint64_t source = 0;
  uint64_t bytes = 0;
  uint64_t signalOffset = 0;
  int64_t value = 0;
  uint64_t budget = 0;
  void put(int remotePeer, uint64_t dst, uint64_t src, uint64_t size) {
    peer = remotePeer;
    destination = dst;
    source = src;
    bytes = size;
    ++puts;
  }
  void putWithSignal(int remotePeer, uint64_t dst, uint64_t src, uint64_t size, uint64_t signal, uint64_t add) {
    put(remotePeer, dst, src, size);
    signalOffset = signal;
    value = static_cast<int64_t>(add);
  }
  void atomicAdd(int remotePeer, uint64_t dst, int64_t add) {
    peer = remotePeer;
    destination = dst;
    value = add;
    ++atomics;
  }
  void flush(int remotePeer) {
    peer = remotePeer;
    ++flushes;
  }
  int tryFlush(int remotePeer, uint64_t spins) {
    peer = remotePeer;
    budget = spins;
    ++boundedFlushes;
    return status;
  }
};
}  // namespace mscclpp

#include <mscclpp/port_channel_device.hpp>

int main() {
  using namespace mscclpp;
  uint64_t inbound = 0;
  uint64_t expected = 0;
  uint64_t flushDone = UINT64_MAX;
  std::vector<ProxyTrigger> triggers;
  PortChannelDeviceHandle proxy(12, {&inbound, &expected}, {&triggers}, 9, 7, &flushDone);
  require(proxy.backend_ == PortChannelBackend::Proxy, "proxy is not the default");
  proxy.put(8, 16, 32);
  proxy.signal();
  proxy.putWithSignal(24, 40, 64);
  proxy.putWithSignalAndFlush(32, 48, 128, 17);
  proxy.flush();
  const int64_t operand = -((int64_t{1} << 40) + 3);
  proxy.accumulate(56, operand);
  require(triggers.size() == 6, "proxy calls changed");
  const int types[] = {TriggerPut,   TriggerSignal,    TriggerPutWithSignal, TriggerPutWithSignalAndFlush,
                       TriggerFlush, TriggerAccumulate};
  for (size_t index = 0; index < triggers.size(); ++index) {
    require(triggers[index].type == types[index] && triggers[index].semaphoreId == 12, "proxy opcode or semaphore");
  }
  require(triggers[0].dstId == 9 && triggers[0].srcId == 7 && triggers[0].dstOffset == 8 &&
              triggers[0].srcOffset == 16 && triggers[0].size == 32,
          "proxy addressing");
  require((triggers[5].srcOffset << TriggerBitsSize | triggers[5].size) == static_cast<uint64_t>(operand),
          "proxy truncated signed 64-bit accumulate");

  GpuNetIoDeviceContext context;
  PortChannelDeviceHandle network(&context, 3, 64, &inbound, &expected);
  require(network.backend_ == PortChannelBackend::GpuNetIo, "GDAKI selection");
  network.put(8, 16, 32);
  require(context.peer == 3 && context.destination == 8 && context.source == 16 && context.bytes == 32, "GDAKI put");
  network.signal();
  require(context.peer == 3 && context.destination == 64 && context.value == 1, "GDAKI signal address");
  network.putWithSignal(24, 40, 64);
  require(context.signalOffset == 64 && context.value == 1, "GDAKI fused signal");
  network.putWithSignalAndFlush(32, 48, 128, 17);
  require(context.boundedFlushes == 1 && context.budget == 17, "bounded flush removed in release build");
  network.flush(0);
  require(context.boundedFlushes == 2 && context.budget == 0, "zero-budget flush");
  network.flush(-1);
  require(context.flushes == 1, "unbounded flush");
  network.accumulate(56, operand);
  require(context.destination == 56 && context.value == operand, "GDAKI accumulate");
  BasePortChannelDeviceHandle base(&context, 3, 64, &inbound, &expected);
  base.put(99, 8, 77, 16, 32);
  require(context.peer == 3, "proxy memory ID used as GDAKI peer");
  require(!network.poll(), "unsignaled counter");
  inbound = 2;
  require(network.poll(), "signal polling");
  network.wait();
  require(expected == 2 && !network.poll(), "signal consumption");
#ifdef TEST_DEVICE_ASSERTS
  context.status = 16;
  bool rejected = false;
  try {
    network.flush(7);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected && context.budget == 7, "flush failure ignored");
#endif
  std::cout << "PortChannel proxy/GDAKI routing checks passed\n";
}
#endif