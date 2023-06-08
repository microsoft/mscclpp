#ifndef MSCCLPP_CHANNEL_HPP_
#define MSCCLPP_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/epoch.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/proxy.hpp>

namespace mscclpp {
namespace channel {

// A Channel pairs a Connection with an Epoch
class Channel {
 public:
  Channel(Communicator& communicator, std::shared_ptr<Connection> connection)
      : connection_(connection), epoch_(std::make_shared<DeviceEpoch>(communicator, connection)){};

  Connection& connection() { return *connection_; }
  DeviceEpoch& epoch() { return *epoch_; }

 private:
  std::shared_ptr<Connection> connection_;
  std::shared_ptr<DeviceEpoch> epoch_;
};

using ChannelId = uint32_t;

using TriggerType = uint64_t;
const TriggerType TriggerData = 0x1;
const TriggerType TriggerFlag = 0x2;
const TriggerType TriggerSync = 0x4;

// This is just a numeric ID. Each HostConnection will have an internal array indexed by these handles
// mapping to the actual
using MemoryId = uint32_t;

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

union ChannelPacket {
  // Flags have to be *after* data, because otherwise, an incomplete receive from the network may receive the flag but
  // not the data. Note this is assuming that either we receive contiguous chunks of data (sockets) or data is written
  // with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };

  struct {
    uint64_t x;
    uint64_t y;
  } vec;

  uint64_t v[2];

#ifdef __CUDACC__
  __forceinline__ __device__ ChannelPacket() {}
  __forceinline__ __device__ void write(uint32_t val1, uint32_t val2, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"(val1), "r"(flag), "r"(val2), "r"(flag));
  }
  __forceinline__ __device__ void write(uint32_t val1, uint32_t val2) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,1,%2,1};" ::"l"(v), "r"(val1), "r"(val2));
  }
  __forceinline__ __device__ void write(uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"((uint32_t)val), "r"(flag),
                 "r"((uint32_t)(val >> 32)), "r"(flag));
  }
  __forceinline__ __device__ void write(uint64_t val) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,1,%2,1};" ::"l"(v), "r"((uint32_t)val),
                 "r"((uint32_t)(val >> 32)));
  }
  __forceinline__ __device__ uint2 read(uint32_t flag) {
    uint2 data;
    uint32_t flag1, flag2;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                   : "l"(v));
    } while ((flag1 != flag) || (flag2 != flag));
    return data;
  }
  __forceinline__ __device__ uint2 read() { return read(1); }
  __forceinline__ __device__ void clear() {
    vec.x = 0;
    vec.y = 0;
  }
#endif  // __CUDACC__
};

struct DeviceChannel {
  DeviceChannel() = default;

  DeviceChannel(ChannelId channelId, DeviceEpoch::DeviceHandle epoch, DeviceProxyFifo fifo)
      : channelId_(channelId), epoch_(epoch), fifo_(fifo) {}

  DeviceChannel(const DeviceChannel& other) = default;

  DeviceChannel& operator=(DeviceChannel& other) = default;

#ifdef __CUDACC__
  __forceinline__ __device__ void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                      uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, channelId_).value);
  }

  __forceinline__ __device__ void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void putDirect(void* dst, void* src, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                            uint32_t threadId, uint32_t numThreads) {
    // assume the memory is aligned to 8 bytes
    uint64_t* srcAddr = (uint64_t*)((char*)src + srcOffset);
    uint64_t* dstAddr = (uint64_t*)((char*)dst + dstOffset);
    uint64_t ele;
    size_t nElem = size % sizeof(uint64_t) ? (size + sizeof(uint64_t)) / sizeof(uint64_t) : size / sizeof(uint64_t);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      // load to register first
      ele = srcAddr[i];
      dstAddr[i] = ele;
    }
  }

  __forceinline__ __device__ void signalDirect() { epoch_.signalDirect(); }

  __forceinline__ __device__ void signalPacket() { epoch_.signalPacket(); }

  __forceinline__ __device__ void signal() {
    epochIncrement();
    fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, channelId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                uint64_t size) {
    epochIncrement();
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, channelId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src,
                                                        uint64_t srcOffset, uint64_t size) {
    epochIncrement();
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, channelId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, channelId_).value);
    fifo_.sync(curFifoHead);
  }

  __forceinline__ __device__ void wait() { epoch_.wait(); }

  __forceinline__ __device__ void putPacket(void* dst, void* src, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                            uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
    uint32_t* srcBase = (uint32_t*)((char*)src + srcOffset);
    ChannelPacket* dstBase = (ChannelPacket*)((char*)dst + dstOffset);
    size_t nElem = size / sizeof(uint64_t);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      ChannelPacket* pkt = &dstBase[i];
      pkt->write(srcBase[2 * i], srcBase[2 * i + 1], flag);
    }
  }

  __forceinline__ __device__ void putPacket(void* dst, void* src, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                            uint32_t threadId, uint32_t numThreads) {
    // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
    uint32_t* srcBase = (uint32_t*)((char*)src + srcOffset);
    ChannelPacket* dstBase = (ChannelPacket*)((char*)dst + dstOffset);
    size_t nElem = size / sizeof(uint64_t);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      ChannelPacket* pkt = &dstBase[i];
      pkt->write(srcBase[2 * i], srcBase[2 * i + 1]);
    }
  }

  __forceinline__ __device__ void getPacket(void* dst, void* src, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                            uint32_t threadId, uint32_t numThreads, uint32_t flag) {
    // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
    ChannelPacket* srcBase = (ChannelPacket*)((char*)src + srcOffset);
    uint2* dstBase = (uint2*)((char*)dst + dstOffset);
    size_t nElem = size / sizeof(uint2);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      ChannelPacket* pkt = &srcBase[i];
      dstBase[i] = pkt->read(flag);
      // for future reuse
      pkt->clear();
    }
  }

  __forceinline__ __device__ void epochIncrement() { epoch_.epochIncrement(); }

  __forceinline__ __device__ uint64_t epochGetLocal() const { return epoch_.epochGetLocal(); }
#endif  // __CUDACC__

  ChannelId channelId_;

  DeviceEpoch::DeviceHandle epoch_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo_;
};

class DeviceChannelService;

inline ProxyHandler makeChannelProxyHandler(DeviceChannelService& channelService);

class DeviceChannelService {
 public:
  DeviceChannelService(Communicator& communicator);

  ChannelId addChannel(std::shared_ptr<Connection> connection) {
    channels_.push_back(Channel(communicator_, connection));
    return channels_.size() - 1;
  }

  MemoryId addMemory(RegisteredMemory memory) {
    memories_.push_back(memory);
    return memories_.size() - 1;
  }

  Channel channel(ChannelId id) { return channels_[id]; }
  DeviceChannel deviceChannel(ChannelId id) {
    return DeviceChannel(id, channels_[id].epoch().deviceHandle(), proxy_.fifo().deviceFifo());
  }

  void startProxy() { proxy_.start(); }
  void stopProxy() { proxy_.stop(); }

 private:
  Communicator& communicator_;
  std::vector<Channel> channels_;
  std::vector<RegisteredMemory> memories_;
  Proxy proxy_;
  int deviceNumaNode;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw) {
    ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
    Channel& channel = channels_[trigger->fields.chanId];

    auto result = ProxyHandlerResult::Continue;

    if (trigger->fields.type & TriggerData) {
      RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
      RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
      channel.connection().write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset, trigger->fields.size);
    }

    if (trigger->fields.type & TriggerFlag) {
      channel.epoch().signal();
    }

    if (trigger->fields.type & TriggerSync) {
      channel.connection().flush();
      result = ProxyHandlerResult::FlushFifoTailAndContinue;
    }

    return result;
  }
};

struct SimpleDeviceChannel {
  SimpleDeviceChannel() = default;

  SimpleDeviceChannel(DeviceChannel devChan, MemoryId dst, MemoryId src) : devChan_(devChan), dst_(dst), src_(src) {}

  SimpleDeviceChannel(DeviceChannel devChan, void* dstPtr, void* srcPtr, void* tmpPtr = nullptr)
      : devChan_(devChan), dstPtr_(dstPtr), srcPtr_(srcPtr), tmpPtr_(tmpPtr) {}

  SimpleDeviceChannel(DeviceChannel devChan, MemoryId dst, MemoryId src, void* dstPtr, void* srcPtr,
                      void* tmpPtr = nullptr)
      : devChan_(devChan), dst_(dst), src_(src), dstPtr_(dstPtr), srcPtr_(srcPtr), tmpPtr_(tmpPtr) {}

  SimpleDeviceChannel(const SimpleDeviceChannel& other) = default;

  SimpleDeviceChannel& operator=(SimpleDeviceChannel& other) = default;

#ifdef __CUDACC__

  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    devChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  __forceinline__ __device__ void putDirect(uint64_t offset, uint64_t size, uint32_t threadId, uint32_t numThreads) {
    devChan_.putDirect(dstPtr_, srcPtr_, offset, offset, size, threadId, numThreads);
  }

  __forceinline__ __device__ void signal() { devChan_.signal(); }

  __forceinline__ __device__ void signalDirect() { devChan_.signalDirect(); }

  __forceinline__ __device__ void signalPacket() { devChan_.signalPacket(); }

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

  __forceinline__ __device__ void putPacket(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                            uint32_t numThreads, uint32_t flag) {
    devChan_.putPacket(dstPtr_, srcPtr_, dstOffset, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void putPacket(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                            uint32_t numThreads) {
    devChan_.putPacket(dstPtr_, srcPtr_, dstOffset, srcOffset, size, threadId, numThreads);
  }

  __forceinline__ __device__ void getPacket(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                            uint32_t numThreads, uint32_t flag) {
    devChan_.getPacket(srcPtr_, tmpPtr_, dstOffset, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void epochIncrement() { devChan_.epochIncrement(); }

  __forceinline__ __device__ uint64_t epochGetLocal() const { return devChan_.epochGetLocal(); }

#endif  // __CUDACC__

  DeviceChannel devChan_;
  MemoryId dst_;
  MemoryId src_;

  // these are used for direct copy
  void* dstPtr_;
  void* srcPtr_;

  // extra local buffer for out-of-place copy
  void* tmpPtr_;
};

}  // namespace channel
}  // namespace mscclpp

#endif  // MSCCLPP_CHANNEL_HPP_
