// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include <type_traits>

#include "detail/gpunetio_device.hpp"
#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Where a PortChannel's operations are serviced.
enum class PortChannelBackend : uint8_t { Host = 0, Device = 1 };
/// Semaphore ID in a ProxyService registry.
using SemaphoreId = uint32_t;
/// Registered memory ID in a ProxyService registry.
using MemoryId = uint32_t;

struct BasePortChannel;
struct PortChannel;
namespace detail {
struct PortChannelHandleAccess;
}

/// Backend is the only handle specialization axis. Concurrency and scratch are operation policies.
template <PortChannelBackend Backend = PortChannelBackend::Host>
struct BasePortChannelDeviceHandleT;
template <PortChannelBackend Backend = PortChannelBackend::Host>
struct PortChannelDeviceHandleT;

/// Host-proxy device view without bound memory regions.
template <>
struct BasePortChannelDeviceHandleT<PortChannelBackend::Host> {
 public:
  /// Scope of concurrent callers on the underlying queue pair.
  enum class Concurrency : uint8_t { Exclusive = 0, Block = 1, Device = 2 };
  /// Default scratch capacity per CTA (unused by Host operations).
  static constexpr uint32_t kDefaultSmemSlots = detail::kPortChannelDefaultSmemSlots;

  /// Default view. Operations fail explicitly until valid state is assigned.
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandleT() : receive_{}, host_{} {}
  /// Construct a Host-proxy handle. Prefer extraction from a host channel.
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandleT(SemaphoreId id,
                                                                    Host2DeviceSemaphoreDeviceHandle receive,
                                                                    FifoDeviceHandle fifo, uint64_t* flushDone)
      : receive_(receive), host_{id, fifo, flushDone} {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put between ProxyService memory IDs. Only Concurrency::Device is supported.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    validate<Mode, SmemSlots>();
    host_.fifo.push({TriggerPut, dst, dstOffset, src, srcOffset, size, host_.id});
  }
  /// Put between ProxyService memory IDs at the same offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put<Mode, SmemSlots>(dst, offset, src, offset, size);
  }
  /// Signal the remote peer through the proxy.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void signal() {
    validate<Mode, SmemSlots>();
    host_.fifo.push({TriggerSignal, 0, 0, 0, 0, 0, host_.id});
  }
  /// Put and signal in FIFO order.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                           uint64_t size) {
    validate<Mode, SmemSlots>();
    host_.fifo.push({TriggerPutWithSignal, dst, dstOffset, src, srcOffset, size, host_.id});
  }
  /// Put and signal at a common offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal<Mode, SmemSlots>(dst, offset, src, offset, size);
  }
  /// Put, signal and wait for local completion with one FIFO trigger.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                   uint64_t size, int64_t maxSpinCount = 1000000) {
    validate<Mode, SmemSlots>();
    auto pos = host_.fifo.push({TriggerPutWithSignalAndFlush, dst, dstOffset, src, srcOffset, size, host_.id});
    waitFlush(pos, maxSpinCount);
  }
  /// Put, signal and flush at a common offset.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush<Mode, SmemSlots>(dst, offset, src, offset, size, maxSpinCount);
  }
  /// Wait for local completion of operations posted so far on this connection.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    validate<Mode, SmemSlots>();
    auto pos = host_.fifo.push({TriggerFlush, 0, 0, 0, 0, 0, host_.id});
    waitFlush(pos, maxSpinCount);
  }
  /// Add to a Host-registered destination. See Connection::accumulate for transport support.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void accumulate(MemoryId dst, uint64_t offset, int64_t value) {
    validate<Mode, SmemSlots>();
    uint64_t operand = static_cast<uint64_t>(value);
    host_.fifo.push(
        {TriggerAccumulate, dst, offset, 0, operand >> TriggerBitsSize, static_cast<uint32_t>(operand), host_.id});
  }
  /// Poll the shared receive counter, consuming one expected signal when available.
  MSCCLPP_DEVICE_INLINE bool poll() {
    requireReceive();
    return receive_.poll();
  }
  /// Wait for a remote signal.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    requireReceive();
    receive_.wait(maxSpinCount);
  }
#endif

 private:
  struct HostState {
    SemaphoreId id;
    FifoDeviceHandle fifo;
    uint64_t* flushDone;
  };
  Host2DeviceSemaphoreDeviceHandle receive_;
  HostState host_;
#if defined(MSCCLPP_DEVICE_COMPILE)
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void validate() const {
    static_assert(Mode == Concurrency::Device, "Host channels only support Concurrency::Device");
    static_assert(SmemSlots <= 1024, "At most 1024 shared-memory slots per CTA are supported");
    detail::portChannelRequire(host_.fifo.head != nullptr, "unassigned channel handle");
  }
  MSCCLPP_DEVICE_INLINE void requireReceive() const {
    detail::portChannelRequire(receive_.inboundToken && receive_.expectedInboundToken,
                               "unassigned channel receive state");
  }
  MSCCLPP_DEVICE_INLINE void waitFlush(uint64_t pos, [[maybe_unused]] int64_t maxSpinCount) {
    detail::portChannelRequire(host_.flushDone != nullptr, "missing Host flush state");
    POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(host_.flushDone, memoryOrderAcquire) <= pos), maxSpinCount);
  }
#endif
  friend struct BasePortChannel;
  friend struct PortChannel;
  friend struct detail::PortChannelHandleAccess;
};

/// Legacy Host-proxy view without bound memory regions.
using BasePortChannelDeviceHandle = BasePortChannelDeviceHandleT<PortChannelBackend::Host>;

/// Direct GPU-posting device view. No ProxyService memory-ID operations are available.
template <>
struct BasePortChannelDeviceHandleT<PortChannelBackend::Device> {
 public:
  /// Scope of concurrent callers on the underlying queue pair.
  using Concurrency = BasePortChannelDeviceHandle::Concurrency;
  /// Default scratch capacity per CTA. Select capacity with each operation's second template argument.
  static constexpr uint32_t kDefaultSmemSlots = detail::kPortChannelDefaultSmemSlots;
  /// Default view. Operations fail explicitly until valid state is assigned.
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandleT() : receive_{}, device_{} {}
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Signal the remote peer. slotIdx=-1 selects the linear CTA thread index for scratch.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void signal(int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    device_.backend.template accumulate<Mode, SmemSlots>(device_.remoteSignal, 0, 1, slotIdx);
  }
  /// Wait for local completion of operations posted so far on this connection.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    validate<Mode, SmemSlots>();
    device_.backend.template flush<Mode>(maxSpinCount);
  }
  /// Poll the shared receive counter, consuming one expected signal when available.
  MSCCLPP_DEVICE_INLINE bool poll() {
    requireReceive();
    return receive_.poll();
  }
  /// Wait for a remote signal.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    requireReceive();
    receive_.wait(maxSpinCount);
  }
#endif
 private:
  struct DeviceState {
    detail::GpuNetIoDeviceHandle backend;
    detail::GpuNetIoMemoryDeviceHandle remoteSignal;
  };
  Host2DeviceSemaphoreDeviceHandle receive_;
  DeviceState device_;
  MSCCLPP_HOST_DEVICE_INLINE constexpr BasePortChannelDeviceHandleT(Host2DeviceSemaphoreDeviceHandle receive,
                                                                    detail::GpuNetIoDeviceHandle backend,
                                                                    detail::GpuNetIoMemoryDeviceHandle remoteSignal)
      : receive_(receive), device_{backend, remoteSignal} {}
#if defined(MSCCLPP_DEVICE_COMPILE)
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void validate() const {
    static_assert(SmemSlots <= 1024, "At most 1024 shared-memory slots per CTA are supported");
    static_assert(Mode == Concurrency::Exclusive || Mode == Concurrency::Block || Mode == Concurrency::Device);
    requireReceive();
    detail::portChannelRequire(device_.backend.qp != nullptr, "unassigned Device posting state");
  }
  MSCCLPP_DEVICE_INLINE void requireReceive() const {
#if !MSCCLPP_DETAIL_GPUNETIO_DEVICE_AVAILABLE
    detail::portChannelRequire(false, "Device backend is not available for this build/architecture");
#endif
    detail::portChannelRequire(receive_.inboundToken && receive_.expectedInboundToken,
                               "unassigned channel receive state");
  }
#endif
  friend struct BasePortChannel;
  friend struct PortChannel;
  friend struct PortChannelDeviceHandleT<PortChannelBackend::Device>;
  friend struct detail::PortChannelHandleAccess;
};

/// Bound Host-proxy device view. Normal names continue to select this backend.
template <>
struct PortChannelDeviceHandleT<PortChannelBackend::Host> : public BasePortChannelDeviceHandle {
 public:
  /// Default view. Operations fail explicitly until valid state is assigned.
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandleT()
      : BasePortChannelDeviceHandle(), boundsInitialized_(false), bound_{} {}
  /// Construct a bound Host-proxy view. Prefer host-channel extraction.
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandleT(SemaphoreId id,
                                                                Host2DeviceSemaphoreDeviceHandle receive,
                                                                FifoDeviceHandle fifo, MemoryId dst, MemoryId src,
                                                                uint64_t* flushDone)
      : BasePortChannelDeviceHandle(id, receive, fifo, flushDone), boundsInitialized_(true), bound_{dst, src} {}
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put between the bound ProxyService memory regions.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    requireBounds();
    BasePortChannelDeviceHandle::put<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset, size);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put between bound regions and signal the remote peer.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    requireBounds();
    BasePortChannelDeviceHandle::putWithSignal<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset, size);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put, signal and wait for local completion.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    requireBounds();
    BasePortChannelDeviceHandle::putWithSignalAndFlush<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset,
                                                                        size, maxSpinCount);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode, SmemSlots>(offset, offset, size);
  }
  /// Add to the bound destination region.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t offset, int64_t value) {
    requireBounds();
    BasePortChannelDeviceHandle::accumulate<Mode, SmemSlots>(bound_.dst, offset, value);
  }
#endif
 private:
  struct HostBound {
    MemoryId dst;
    MemoryId src;
  };
  // Base-subobject assignment must not initialize bound IDs; zero is a valid registry ID.
  bool boundsInitialized_;
  HostBound bound_;
  MSCCLPP_HOST_DEVICE_INLINE void requireBounds() const {
    detail::portChannelRequire(boundsInitialized_, "unassigned Host bound regions");
  }
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandleT(BasePortChannelDeviceHandle base, HostBound bound)
      : BasePortChannelDeviceHandle(base), boundsInitialized_(true), bound_(bound) {}
  friend struct PortChannel;
  friend struct detail::PortChannelHandleAccess;
};

/// Legacy bound Host-proxy device view.
using PortChannelDeviceHandle = PortChannelDeviceHandleT<PortChannelBackend::Host>;

/// Bound direct GPU-posting view. Descriptors remain private to channel extraction.
template <>
struct PortChannelDeviceHandleT<PortChannelBackend::Device>
    : public BasePortChannelDeviceHandleT<PortChannelBackend::Device> {
 public:
  /// Default view. Operations fail explicitly until valid state is assigned.
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandleT() : BasePortChannelDeviceHandleT(), bound_{} {}
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put between bound registered regions. slotIdx=-1 selects the linear CTA thread index for scratch.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    device_.backend.template put<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset, size, slotIdx);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put between bound regions and signal the remote peer.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    device_.backend.template putWithAccumulate<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset, size,
                                                                device_.remoteSignal, 0, 1, slotIdx);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode, SmemSlots>(offset, offset, size);
  }
  /// Put, signal and wait for local completion.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000, int slotIdx = -1) {
    putWithSignal<Mode, SmemSlots>(dstOffset, srcOffset, size, slotIdx);
    flush<Mode, SmemSlots>(maxSpinCount);
  }
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode, SmemSlots>(offset, offset, size);
  }
  /// Atomically add to the bound destination region.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t offset, int64_t value, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    device_.backend.template accumulate<Mode, SmemSlots>(bound_.dst, offset, value, slotIdx);
  }
  /// Post a put followed by an atomic add. Observing the add publishes the put.
  template <Concurrency Mode = Concurrency::Device, uint32_t SmemSlots = kDefaultSmemSlots>
  MSCCLPP_DEVICE_INLINE void putWithAccumulate(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                               uint64_t accOffset, int64_t value, int slotIdx = -1) {
    validate<Mode, SmemSlots>();
    device_.backend.template putWithAccumulate<Mode, SmemSlots>(bound_.dst, dstOffset, bound_.src, srcOffset, size,
                                                                bound_.dst, accOffset, value, slotIdx);
  }
#endif
 private:
  struct DeviceBound {
    detail::GpuNetIoMemoryDeviceHandle dst;
    detail::GpuNetIoMemoryDeviceHandle src;
  };
  DeviceBound bound_;
#if defined(MSCCLPP_DEVICE_COMPILE)
  template <Concurrency Mode, uint32_t SmemSlots>
  MSCCLPP_DEVICE_INLINE void validate() const {
    BasePortChannelDeviceHandleT::template validate<Mode, SmemSlots>();
    detail::portChannelRequire(bound_.dst.addr != 0 && bound_.src.addr != 0, "unassigned Device bound regions");
  }
#endif
  MSCCLPP_HOST_DEVICE_INLINE constexpr PortChannelDeviceHandleT(BasePortChannelDeviceHandleT base, DeviceBound bound)
      : BasePortChannelDeviceHandleT(base), bound_(bound) {}
  friend struct PortChannel;
  friend struct detail::PortChannelHandleAccess;
};

static_assert(std::is_trivially_copyable_v<BasePortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<PortChannelDeviceHandle>);
static_assert(std::is_trivially_copyable_v<BasePortChannelDeviceHandleT<PortChannelBackend::Device>>);
static_assert(std::is_trivially_copyable_v<PortChannelDeviceHandleT<PortChannelBackend::Device>>);

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
