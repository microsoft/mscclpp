#ifndef MSCCLPP_CHANNEL_HPP_
#define MSCCLPP_CHANNEL_HPP_

#include "epoch.hpp"
#include "mscclpp.hpp"
#include "mscclppfifo.hpp"
#include "proxy.hpp"
#include "utils.hpp"

namespace mscclpp {
namespace channel {

// A Channel pairs a Connection with an Epoch
class Channel
{
public:
  Channel(Communicator& communicator, std::shared_ptr<Connection> connection)
    : connection_(connection), epoch_(std::make_shared<Epoch>(communicator, connection)){};

  Connection& connection()
  {
    return *connection_;
  }
  Epoch& epoch()
  {
    return *epoch_;
  }

private:
  std::shared_ptr<Connection> connection_;
  std::shared_ptr<Epoch> epoch_;
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
  struct
  {
    // first 64 bits: value[0]
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET); // ensure 64-bit alignment
    // second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE -
                MSCCLPP_BITS_TYPE); // ensure 64-bit alignment
  } fields;

#ifdef __CUDACC__
  __device__ ChannelTrigger()
  {
  }
  __device__ ChannelTrigger(ProxyTrigger value) : value(value)
  {
  }
  __device__ ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                            uint64_t size, int connectionId)
  {
    value.fst = ((srcOffset << MSCCLPP_BITS_SIZE) + size);
    value.snd = ((((((((connectionId << MSCCLPP_BITS_TYPE) + (uint64_t)type) << MSCCLPP_BITS_REGMEM_HANDLE) + dst)
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   src)
                  << MSCCLPP_BITS_OFFSET) +
                 dstOffset);
  }
#endif // __CUDACC__
};

struct DeviceChannel
{
  DeviceChannel() = default;

  DeviceChannel(ChannelId channelId, DeviceEpoch epoch, DeviceProxyFifo fifo)
    : channelId_(channelId), epoch_(epoch), fifo_(fifo)
  {
  }

  DeviceChannel(const DeviceChannel& other) = default;

  DeviceChannel& operator=(DeviceChannel& other) = default;

#ifdef __CUDACC__
  __forceinline__ __device__ void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size)
  {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, channelId_).value);
  }

  __forceinline__ __device__ void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size)
  {
    put(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void signal()
  {
    epochIncrement();
    fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, channelId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                uint64_t size)
  {
    epochIncrement();
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, channelId_).value);
  }

  __forceinline__ __device__ void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size)
  {
    putWithSignal(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src,
                                                        uint64_t srcOffset, uint64_t size)
  {
    epochIncrement();
    uint64_t curFifoHead = fifo_.push(
      ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, channelId_).value);
    while (*(volatile uint64_t*)&fifo_.triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo_.tailReplica <= curFifoHead)
      ;
  }

  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size)
  {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void flush()
  {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(mscclppSync, 0, 0, 0, 0, 1, channelId_).value);
    // we need to wait for two conditions to be met to ensure the CPU is done flushing. (1) wait for the tail
    // to go pass by curFifoHead (this is safety net) and (2) wait for the work element value to change to 0.
    while (*(volatile uint64_t*)&fifo_.triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo_.tailReplica <= curFifoHead)
      ;
  }

  __forceinline__ __device__ void wait()
  {
    epoch_.wait();
  }

  __forceinline__ __device__ void epochIncrement()
  {
    epoch_.epochIncrement();
  }
#endif // __CUDACC__

  ChannelId channelId_;

  DeviceEpoch epoch_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo_;
};

class DeviceChannelService;

inline ProxyHandler makeChannelProxyHandler(DeviceChannelService& channelService);

class DeviceChannelService
{
public:
  DeviceChannelService(Communicator& communicator);

  ChannelId addChannel(std::shared_ptr<Connection> connection)
  {
    channels_.push_back(Channel(communicator_, connection));
    return channels_.size() - 1;
  }

  MemoryId addMemory(RegisteredMemory memory)
  {
    memories_.push_back(memory);
    return memories_.size() - 1;
  }

  Channel channel(ChannelId id)
  {
    return channels_[id];
  }
  DeviceChannel deviceChannel(ChannelId id)
  {
    return DeviceChannel(id, channels_[id].epoch().deviceEpoch(), proxy_.fifo().deviceFifo());
  }

  void startProxy()
  {
    proxy_.start();
  }
  void stopProxy()
  {
    proxy_.stop();
  }

private:
  Communicator& communicator_;
  std::vector<Channel> channels_;
  std::vector<RegisteredMemory> memories_;
  Proxy proxy_;
  int deviceNumaNode;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw)
  {
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

struct SimpleDeviceChannel
{
  SimpleDeviceChannel() = default;

  SimpleDeviceChannel(DeviceChannel devChan, MemoryId dst, MemoryId src) : devChan_(devChan), dst_(dst), src_(src)
  {
  }

  SimpleDeviceChannel(const SimpleDeviceChannel& other) = default;

  SimpleDeviceChannel& operator=(SimpleDeviceChannel& other) = default;

#ifdef __CUDACC__

  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void put(uint64_t offset, uint64_t size)
  {
    put(offset, offset, size);
  }

  __forceinline__ __device__ void signal()
  {
    devChan_.signal();
  }

  __forceinline__ __device__ void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t offset, uint64_t size)
  {
    putWithSignal(offset, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size)
  {
    putWithSignalAndFlush(offset, offset, size);
  }

  __forceinline__ __device__ void flush()
  {
    devChan_.flush();
  }

  __forceinline__ __device__ void wait()
  {
    devChan_.wait();
  }

  __forceinline__ __device__ void epochIncrement()
  {
    devChan_.epochIncrement();
  }

#endif // __CUDACC__

  DeviceChannel devChan_;
  MemoryId dst_;
  MemoryId src_;
};

} // namespace channel
} // namespace mscclpp

#endif // MSCCLPP_CHANNEL_HPP_
