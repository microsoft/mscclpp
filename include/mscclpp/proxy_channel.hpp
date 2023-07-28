// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_HPP_
#define MSCCLPP_PROXY_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/semaphore.hpp>

namespace mscclpp {

using SemaphoreId = uint32_t;

/// Numeric ID of @ref RegisteredMemory. @ref ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

struct ProxyChannel;

/// Base class for proxy services. Proxy services are used to proxy data between devices.
class BaseProxyService {
 public:
  BaseProxyService() = default;
  virtual ~BaseProxyService() = default;
  virtual void startProxy() = 0;
  virtual void stopProxy() = 0;
};

/// Proxy service implementation.
class ProxyService : public BaseProxyService {
 public:
  /// Constructor.
  /// @param communicator The communicator to use.
  ProxyService(Communicator& communicator);

  /// Add a semaphore to the proxy service.
  /// @param connection The connection associated with the semaphore.
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(std::shared_ptr<Connection> connection);

  /// Add a 2D channel to the proxy service.
  /// @param connection The connection associated with the channel.
  /// @param pitch The pitch pair.
  SemaphoreId add2DChannel(std::shared_ptr<Connection> connection, std::pair<uint64_t, uint64_t> pitch);

  /// Register a memory region with the proxy service.
  /// @param memory The memory region to register.
  /// @return The ID of the memory region.
  MemoryId addMemory(RegisteredMemory memory);

  /// Get a semaphore by ID.
  /// @param id The ID of the semaphore.
  /// @return The semaphore.
  std::shared_ptr<Host2DeviceSemaphore> semaphore(SemaphoreId id) const;

  /// Get a proxy channel by semaphore ID.
  /// @param id The ID of the semaphore.
  /// @return The proxy channel.
  ProxyChannel deviceChannel(SemaphoreId id);

  /// Start the proxy service.
  void startProxy();

  /// Stop the proxy service.
  void stopProxy();

 private:
  Communicator& communicator_;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::vector<std::pair<uint64_t, uint64_t>> pitches_;
  Proxy proxy_;
  int deviceNumaNode;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
};

using TriggerType = uint64_t;
const TriggerType TriggerData = 0x1;  // Trigger a data transfer.
const TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
const TriggerType TriggerSync = 0x4;  // Trigger a flush.

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_REGMEM_HANDLE 8
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10

#define MSCCLPP_BITS_WIDTH_SIZE 16
#define MSCCLPP_BITS_HEIGHT_SIZE 16
#define MSCCLPP_2D_FLAG 1

/// Basic structure of each work element in the FIFO.
union ChannelTrigger {
  ProxyTrigger value;
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_TYPE -
                MSCCLPP_BITS_CONNID);  // ensure 64-bit alignment
  } fields;

  struct {
    // First 64 bits: value[0]
    uint64_t width : MSCCLPP_BITS_WIDTH_SIZE;
    uint64_t height : MSCCLPP_BITS_HEIGHT_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t
        : (64 - MSCCLPP_BITS_WIDTH_SIZE - MSCCLPP_BITS_HEIGHT_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t multiDimensionFlag : MSCCLPP_2D_FLAG;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_TYPE -
                MSCCLPP_BITS_CONNID - MSCCLPP_2D_FLAG);  // ensure 64-bit alignment
  } fields2D;

#ifdef __CUDACC__
  /// Default constructor.
  __device__ ChannelTrigger() {}

  /// Copy constructor.
  __device__ ChannelTrigger(ProxyTrigger value) : value(value) {}

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  __device__ ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                            uint64_t bytes, int semaphoreId) {
    value.fst = ((srcOffset << MSCCLPP_BITS_SIZE) + bytes);
    value.snd = ((((((((semaphoreId << MSCCLPP_BITS_TYPE) + (uint64_t)type) << MSCCLPP_BITS_REGMEM_HANDLE) + dst)
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   src)
                  << MSCCLPP_BITS_OFFSET) +
                 dstOffset);
  }

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  /// @param semaphoreId The ID of the semaphore.
  __device__ ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                            uint64_t width, uint64_t height, int semaphoreId) {
    value.fst = (((srcOffset << MSCCLPP_BITS_HEIGHT_SIZE) + height) << MSCCLPP_BITS_WIDTH_SIZE) + width;
    value.snd = ((((((((((1ULL << MSCCLPP_BITS_CONNID) + semaphoreId) << MSCCLPP_BITS_TYPE) + type)
                      << MSCCLPP_BITS_REGMEM_HANDLE) +
                     dst)
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   src)
                  << MSCCLPP_BITS_OFFSET) +
                 dstOffset);
  }
#endif  // __CUDACC__
};

/// Proxy channel.
struct ProxyChannel {
  // Use DeviceHandle<ProxyChannel> in device code.
  typedef ProxyChannel DeviceHandle;

  ProxyChannel() = default;

  ProxyChannel(SemaphoreId semaphoreId, Host2DeviceSemaphore::DeviceHandle semaphore, DeviceProxyFifo fifo);

  ProxyChannel(const ProxyChannel& other) = default;

  ProxyChannel& operator=(ProxyChannel& other) = default;

#ifdef __CUDACC__
  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                      uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  /// @brief Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2D(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                        uint32_t width, uint32_t height) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, width, height, semaphoreId_).value);
  }

  /// @brief Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2D(MemoryId dst, MemoryId src, uint64_t offset, uint32_t width, uint32_t height) {
    put2D(dst, offset, src, offset, width, height);
  }

  /// Push a @ref TriggerFlag to the FIFO.
  __forceinline__ __device__ void signal() {
    fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2DWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                  uint32_t width, uint32_t height) {
    fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, width, height, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2DWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint32_t width,
                                                  uint32_t height) {
    put2DWithSignal(dst, offset, src, offset, width, height);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src,
                                                        uint64_t srcOffset, uint64_t size) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  __forceinline__ __device__ void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  /// Wait for the proxy channel to be signaled.
  __forceinline__ __device__ void wait() { semaphore_.wait(); }

#endif  // __CUDACC__

  SemaphoreId semaphoreId_;

  Host2DeviceSemaphore::DeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo_;
};

/// Simple proxy channel with a single destination and source memory region.
struct SimpleProxyChannel {
  // Use DeviceHandle<SimpleProxyChannel> in device code.
  typedef SimpleProxyChannel DeviceHandle;

  /// Default constructor.
  SimpleProxyChannel() = default;

  /// Constructor.
  /// @param proxyChan The proxy channel.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  SimpleProxyChannel(ProxyChannel proxyChan, MemoryId dst, MemoryId src);

  /// Constructor.
  /// @param proxyChan The proxy channel.
  SimpleProxyChannel(ProxyChannel proxyChan) : proxyChan_(proxyChan) {}

  /// Copy constructor.
  SimpleProxyChannel(const SimpleProxyChannel& other) = default;

  /// Assignment operator.
  SimpleProxyChannel& operator=(SimpleProxyChannel& other) = default;

#ifdef __CUDACC__
  /// Push a @ref TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2D(uint64_t dstOffset, uint64_t srcOffset, uint32_t width, uint32_t height) {
    proxyChan_.put2D(dst_, dstOffset, src_, srcOffset, width, height);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  /// Push a @ref TriggerFlag to the FIFO.
  __forceinline__ __device__ void signal() { proxyChan_.signal(); }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2DWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint32_t width,
                                                  uint32_t height) {
    proxyChan_.put2DWithSignal(dst_, dstOffset, src_, srcOffset, width, height);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param width The width of the 2D region.
  /// @param height The height of the 2D region.
  __forceinline__ __device__ void put2DWithSignal(uint64_t offset, uint32_t width, uint32_t height) {
    put2DWithSignal(offset, offset, width, height);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  __forceinline__ __device__ void flush() { proxyChan_.flush(); }

  /// Wait for the proxy channel to be signaled.
  __forceinline__ __device__ void wait() { proxyChan_.wait(); }

#endif  // __CUDACC__

  ProxyChannel proxyChan_;
  MemoryId dst_;
  MemoryId src_;
};

template <>
DeviceHandle<ProxyChannel> deviceHandle(ProxyChannel&& proxyChannel);

template <>
DeviceHandle<SimpleProxyChannel> deviceHandle(SimpleProxyChannel&& simpleProxyChannel);
}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_HPP_
