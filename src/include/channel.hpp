#ifndef MSCCLPP_CHANNEL_HPP_
#define MSCCLPP_CHANNEL_HPP_

#include "mscclpp.hpp"
#include "proxy.hpp"

namespace mscclpp {

// For every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER, a flush of the tail to device memory is triggered.
// As long as MSCCLPP_PROXY_FIFO_SIZE is large enough, having a stale tail is not a problem.
#define MSCCLPP_PROXY_FIFO_SIZE 128
#define MSCCLPP_PROXY_FIFO_FLUSH_COUNTER 4

using ChannelTriggerType = uint64_t;
const ChannelTriggerType channelTriggerData = 0x1;
const ChannelTriggerType channelTriggerFlag = 0x2;
const ChannelTriggerType channelTriggerSync = 0x4;

// This is just a numeric ID. Each HostConnection will have an internal array indexed by these handles
// mapping to the actual 
using BufferHandle = uint32_t;

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_BUFFER_HANDLE 8
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
    uint64_t srcBufferHandle : MSCCLPP_BITS_BUFFER_HANDLE;
    uint64_t dstBufferHandle : MSCCLPP_BITS_BUFFER_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t connId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_BUFFER_HANDLE - MSCCLPP_BITS_BUFFER_HANDLE - MSCCLPP_BITS_TYPE); // ensure 64-bit alignment
  } fields;

#ifdef __CUDACC__
  __device__ ChannelTrigger() {}
  __device__ ChannelTrigger(ProxyTrigger value) : value(value) {}
  __device__ ChannelTrigger(ChannelTriggerType type, BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size, int connectionId) {
    value.fst = ((srcOffset << MSCCLPP_BITS_SIZE) + size);
    value.snd = ((((((((connectionId << MSCCLPP_BITS_TYPE) + (uint64_t)type) << MSCCLPP_BITS_BUFFER_HANDLE) + dst) << MSCCLPP_BITS_BUFFER_HANDLE) + src) << MSCCLPP_BITS_OFFSET) + dstOffset);
  }
#endif // __CUDACC__
};

struct ConnectionEpoch {
#ifdef __CUDACC__
  __forceinline__ __device__ void wait()
  {
    (*waitEpochId) += 1;
    while (*(volatile uint64_t*)&(localSignalEpochId->proxy) < (*waitEpochId))
      ;
  }

  __forceinline__ __device__ void epochIncrement()
  {
    *(volatile uint64_t*)&(localSignalEpochId->device) += 1;
  }
#endif // __CUDACC__

  SignalEpochId* localSignalEpochId;
  // used by the signal() function directly from gpu
  SignalEpochId* remoteSignalEpochId;

  // every wait(), increments this and then the gpu waits for either:
  // 1) localSignalEpochId->proxy to be >= this in case of a proxy thread
  // 2) remoteSignalEpochId->device to be >= this in case of a gpu thread
  uint64_t* waitEpochId;
};

class HostConnection {
  struct Impl;
public:
  /* HostConnection can not be constructed from user code and must instead be created through Communicator::connect */
  HostConnection(std::unique_ptr<Impl>);

  ~HostConnection();

  void write()

  int getId();

  /* Get the number of times registerBuffer(...) was called.
   *
   * Returns: the number of buffers registered
   */
  int numLocalBuffers();

  /* Get the BufferHandle returned by a call to registerBuffer(...) as identified by the index
   *
   * Inputs:
   *  index: the index of the handle to get
   * 
   * Returns: a handle to the buffer
   */
  BufferHandle getLocalBuffer(int index);

  /* Get the number of times registerBuffer(...) was called on the remote peer.
   *
   * Returns: the number of buffers registered on the remote peer
   */
  int numRemoteBuffers();

  /* Get the BufferHandle returned by a call to registerBuffer(...) on the remote peer as identified by the index
   *
   * Inputs:
   *  index: the index of the handle to get
   * 
   * Returns: a handle to the buffer on the remote peer
   */
  BufferHandle getRemoteBuffer(int index);

  ConnectionEpoch getEpoch();

  DeviceProxyFifo getDeviceFifo();

  void put(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size);

  void signal();

  void flush();

  void wait();

private:
  std::unique_ptr<Impl> pimpl;
  friend class Communicator;
};

struct DeviceConnection {
  DeviceConnection() = default;

  DeviceConnection(HostConnection& hostConn)
    : connectionId(hostConn.getId()), epoch(hostConn.getEpoch()),
      fifo(hostConn.getDeviceFifo()) {}

  DeviceConnection(const DeviceConnection& other) = default;

  DeviceConnection& operator=(DeviceConnection& other) = default;

#ifdef __CUDACC__
  __forceinline__ __device__ void put(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size)
  {
    fifo.push(ChannelTrigger(channelTriggerData, dst, dstOffset, src, srcOffset, size, connectionId).value);
  }

  __forceinline__ __device__ void put(BufferHandle dst, BufferHandle src, uint64_t offset, uint64_t size)
  {
    put(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void signal()
  {
    epochIncrement();
    fifo.push(ChannelTrigger(channelTriggerFlag, 0, 0, 0, 0, 1, connectionId).value);
  }

  __forceinline__ __device__ void putWithSignal(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size)
  {
    epochIncrement();
    fifo.push(ChannelTrigger(channelTriggerData | channelTriggerFlag, dst, dstOffset, src, srcOffset, size, connectionId).value);
  }

  __forceinline__ __device__ void putWithSignal(BufferHandle dst, BufferHandle src, uint64_t offset, uint64_t size)
  {
    putWithSignal(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size)
  {
    epochIncrement();
    uint64_t curFifoHead = fifo.push(ChannelTrigger(channelTriggerData | channelTriggerFlag | channelTriggerSync, dst,  dstOffset, src, srcOffset, size, connectionId).value);
    while (*(volatile uint64_t*)&fifo.triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo.tailReplica <= curFifoHead)
      ;
  }

  __forceinline__ __device__ void putWithSignalAndFlush(BufferHandle dst, BufferHandle src, uint64_t offset, uint64_t size)
  {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  __forceinline__ __device__ void flush()
  {
    uint64_t curFifoHead = fifo.push(ChannelTrigger(mscclppSync, 0, 0, 0, 0, 1, connectionId).value);
    // we need to wait for two conditions to be met to ensure the CPU is done flushing. (1) wait for the tail
    // to go pass by curFifoHead (this is safety net) and (2) wait for the work element value to change to 0.
    while (*(volatile uint64_t*)&fifo.triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo.tailReplica <= curFifoHead)
      ;
  }

  __forceinline__ __device__ void wait()
  {
    epoch.wait();
  }

  __forceinline__ __device__ void epochIncrement()
  {
    epoch.epochIncrement();
  }
#endif // __CUDACC__

  int connectionId;

  ConnectionEpoch epoch;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo;
};

struct SimpleDeviceConnection {
  SimpleDeviceConnection() = default;

  SimpleDeviceConnection(HostConnection& hostConn) : devConn(hostConn) {
    dst = hostConn.getRemoteBuffer(0);
    src = hostConn.getLocalBuffer(0);
  }

  SimpleDeviceConnection(const SimpleDeviceConnection& other) = default;

  SimpleDeviceConnection& operator=(SimpleDeviceConnection& other) = default;

#ifdef __CUDACC__

  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devConn.put(dst, dstOffset, src, srcOffset, size);
  }

  __forceinline__ __device__ void put(uint64_t offset, uint64_t size)
  {
    put(offset, offset, size);
  }

  __forceinline__ __device__ void signal()
  {
    devConn.signal();
  }

  __forceinline__ __device__ void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devConn.putWithSignal(dst, dstOffset, src, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t offset, uint64_t size)
  {
    putWithSignal(offset, offset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size)
  {
    devConn.putWithSignalAndFlush(dst, dstOffset, src, srcOffset, size);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size)
  {
    putWithSignalAndFlush(offset, offset, size);
  }

  __forceinline__ __device__ void flush()
  {
    devConn.flush();
  }

  __forceinline__ __device__ void wait()
  {
    devConn.wait();
  }

  __forceinline__ __device__ void epochIncrement()
  {
    devConn.epochIncrement();
  }

#endif // __CUDACC__

  DeviceConnection devConn;
  BufferHandle dst;
  BufferHandle src;
};

