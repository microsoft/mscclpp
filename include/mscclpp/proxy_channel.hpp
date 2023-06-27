// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_HPP_
#define MSCCLPP_PROXY_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/semaphore.hpp>

namespace mscclpp {
namespace channel {

using SemaphoreId = uint32_t;

// This is just a numeric ID. Each HostConnection will have an internal array indexed by these handles
// mapping to the actual
using MemoryId = uint32_t;
struct DeviceChannelHandle;

class BaseProxyService {
 public:
  BaseProxyService() = default;
  virtual ~BaseProxyService() = default;
  virtual void startProxy() = 0;
  virtual void stopProxy() = 0;
};

class ProxyService : public BaseProxyService {
 public:
  ProxyService(Communicator& communicator);

  SemaphoreId addSemaphore(std::shared_ptr<Connection> connection);

  MemoryId addMemory(RegisteredMemory memory);

  std::shared_ptr<Host2DeviceSemaphore> semaphore(SemaphoreId id) const;
  DeviceChannelHandle deviceChannel(SemaphoreId id);

  void startProxy();
  void stopProxy();

 private:
  Communicator& communicator_;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  Proxy proxy_;
  int deviceNumaNode;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
};

using TriggerType = uint64_t;
const TriggerType TriggerData = 0x1;
const TriggerType TriggerFlag = 0x2;
const TriggerType TriggerSync = 0x4;

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_REGMEM_HANDLE 8
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10

// this is the basic structure of each work element in the fifo
// the summation of number of bits must be 128 or less
union ChannelTrigger {
  ProxyTrigger value;
  struct {
    // first 64 bits: value[0]
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    // second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE -
                MSCCLPP_BITS_TYPE);  // ensure 64-bit alignment
  } fields;

#ifdef __CUDACC__
  __device__ ChannelTrigger() {}
  __device__ ChannelTrigger(ProxyTrigger value) : value(value) {}
  __device__ ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                            uint64_t size, int connectionId) {
    value.fst = ((srcOffset << MSCCLPP_BITS_SIZE) + size);
    value.snd = ((((((((connectionId << MSCCLPP_BITS_TYPE) + (uint64_t)type) << MSCCLPP_BITS_REGMEM_HANDLE) + dst)
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   src)
                  << MSCCLPP_BITS_OFFSET) +
                 dstOffset);
  }
#endif  // __CUDACC__
};

struct DeviceChannelHandle {
  DeviceChannelHandle() = default;

  DeviceChannelHandle(SemaphoreId SemaphoreId, Host2DeviceSemaphore::DeviceHandle semaphore, DeviceProxyFifo fifo);

  DeviceChannelHandle(const DeviceChannelHandle& other) = default;

  DeviceChannelHandle& operator=(DeviceChannelHandle& other) = default;

#ifdef __CUDACC__
  __forceinline__ __device__ void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                      uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  __forceinline__ __device__ void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void signal() {
    fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src,
                                                        uint64_t srcOffset, uint64_t size) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  __forceinline__ __device__ void wait() { semaphore_.wait(); }

#endif  // __CUDACC__

  SemaphoreId semaphoreId_;

  Host2DeviceSemaphore::DeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo_;
};

struct SimpleDeviceChannelHandle {
  SimpleDeviceChannelHandle() = default;

  SimpleDeviceChannelHandle(DeviceChannelHandle devChan, MemoryId dst, MemoryId src);

  SimpleDeviceChannelHandle(DeviceChannelHandle devChan) : devChan_(devChan) {}

  SimpleDeviceChannelHandle(const SimpleDeviceChannelHandle& other) = default;

  SimpleDeviceChannelHandle& operator=(SimpleDeviceChannelHandle& other) = default;

#ifdef __CUDACC__
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    devChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  __forceinline__ __device__ void signal() { devChan_.signal(); }

  __forceinline__ __device__ void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    devChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    devChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }

  __forceinline__ __device__ void flush() { devChan_.flush(); }

  __forceinline__ __device__ void wait() { devChan_.wait(); }

#endif  // __CUDACC__

  DeviceChannelHandle devChan_;
  MemoryId dst_;
  MemoryId src_;
};

}  // namespace channel
}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_HPP_
