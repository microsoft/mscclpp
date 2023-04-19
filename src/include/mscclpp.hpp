#ifndef MSCCLPP_HPP_
#define MSCCLPP_HPP_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1
#define MSCCLPP_PATCH 0
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

// For every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER, a flush of the tail to device memory is triggered.
// As long as MSCCLPP_PROXY_FIFO_SIZE is large enough, having a stale tail is not a problem.
#define MSCCLPP_PROXY_FIFO_SIZE 128
#define MSCCLPP_PROXY_FIFO_FLUSH_COUNTER 4

#include <vector>
#include <memory>
#include <functional>

#include <mscclppfifo.hpp>

namespace mscclpp {

struct alignas(16) SignalEpochId {
  // every signal(), increaments this and either:
  // 1) proxy thread pushes it to the remote peer's localSignalEpochId->proxy
  // 2) gpu thread directly writes it to remoteSignalEpochId->device
  uint64_t device;
  // signal() function triggers the cpu proxy thread to write to it
  uint64_t proxy;
};

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

/***************************************************************************************************************
 * A mscclppDevConn provides a zero-copy connection between two GPUs connected via P2P NVLink or InfiniBand.
 * The communication API is one-sided meaning that for every single data transfer, only one side
 * needs to execute unlike a two-sided communication stack such as NCCL where both sides
 * need to execute a send and a receive instruction, respectively, for every transfer.
 *
 * A connection is uniquely identified by the (remoteRank, tag) pair at an endpoint.
 * The two endpoints register buffers of the same size with the connection.
 *
 * The endpoints provide the remoteRank, tag, and the buffer when registering a connection with msccppConnect().
 *
 * mscllppConnectionSetup() sets up all the registered connections.
 *
 ***************************************************************************************************************
 * A proxy thread running on the CPU is necessary to perform transfers using InfiniBand or the DMA engine.
 * The current implementation uses a single proxy thread per context - one IB connection or DMA engine per node.
 * Thus multiple threadblocks using different connections might use the same CPU proxy thread.
 *
 * Before using any of functionality of connections, mscclppProxyLaunch needs to be called to spawn the
 * proxy threads. There are currently two types of connections:
 *
 * P2P via NVLink: the DMA engine can perform the copy between the buffers. DMA engine has higher latency
 * but has a higher bandwidth and costs no compute cycles on the GPU.
 *
 * InfiniBand: the RDMA engine copies the data over MLX devices.
 *
 ***************************************************************************************************************
 * At the runtime, a GPU kernel has access to a mscclppDevConn object that provides the following functions:
 *
 * put(): [non-blocking] the sender initiates a data transfer to the receiver.
 *
 * signal(): [non-blocking] the sender signals the receiver that data is ready to be consumed.
 *
 * flush(): [blocking] the sender waits for all the data transfers to complete
 *
 * wait(): [blocking] the reciever waits on the signal() to start reading the data.
 *
 * The sender should not reuse the buffer till the flush() returns.
 * The receiver should only access the data after the wait() returns.
 *
 * putWithSignal(): the sender initiates a data transfer and signals the receiver that data is ready to be consumed.
 * This is an optimized version of a put() followed by a signal().
 *
 * These functions hide the complexity of syncrhonization between the two GPUs and the CPU proxy thread.
 * Example:
 *
 * // sender GPU
 * devConn.put(data1)
 * // not OK to write to data1
 * devConn.put(data2)
 * // not OK to write to data1, data2
 * devConn.put(data3)                                // receiver GPU
 * // not OK to write to data1, data2, data3         // not OK to read data1, data2, data3
 * devConn.signal() -------------------------------> devConn.wait()
 * // not OK to write to data1, data2, data3         // OK to read data1, data2, data3
 * devConn.flush()
 * // OK to write to data1, data2, data3
 *
 *
 * The two endpoint can concurrently use the same connection provided they are writing (puts) on different
 * indices in the registered buffer.
 **************************************************************************************************************/
struct DeviceConnection {
#ifdef __CUDACC__
  // TODO: add buffer handles

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
    (*waitEpochId) += 1;
    while (*(volatile uint64_t*)&(localSignalEpochId->proxy) < (*waitEpochId))
      ;
  }

  __forceinline__ __device__ void epochIncrement()
  {
    *(volatile uint64_t*)&(localSignalEpochId->device) += 1;
  }

#endif // __CUDACC__

  int connectionId;

  SignalEpochId* localSignalEpochId;
  // used by the signal() function directly from gpu
  SignalEpochId* remoteSignalEpochId;

  // every wait(), increments this and then the gpu waits for either:
  // 1) localSignalEpochId->proxy to be >= this in case of a proxy thread
  // 2) remoteSignalEpochId->device to be >= this in case of a gpu thread
  uint64_t* waitEpochId;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  DeviceProxyFifo fifo;
};

struct SimpleDeviceConnection {
  SimpleDeviceConnection() {}
  SimpleDeviceConnection(DeviceConnection devConn, BufferHandle dst, BufferHandle src) : devConn(devConn), dst(dst), src(src) {}
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

class HostConnection {
  struct Impl;
public:
  /* HostConnection can not be constructed from user code and must instead be created through Communicator::connect */
  HostConnection(std::unique_ptr<Impl>);

  /* Register a region of GPU memory for use with this connection. Must be called before connectionSetup()
   * in the communicator.
   *
   * Inputs:
   *  data: base pointer to the memory
   *  size: size of the memory region in bytes
   * 
   * Returns: a handle to the buffer
   */
  BufferHandle registerBuffer(void* data, uint64_t size);

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

  /* Create a DeviceConnection paired with this HostConnection. A background proxy thread will
   * trigger operations on this HostConnection corresponding to put/signal/etc. calls made to the
   * DeviceConnection.
   * 
   * Returns: the newly created DeviceConnection
   */
  DeviceConnection toDevice();

  void put(BufferHandle dst, uint64_t dstOffset, BufferHandle src, uint64_t srcOffset, uint64_t size);

  void signal();

  void flush();

  void wait();

private:
  std::unique_ptr<Impl> pimpl;
  friend class Communicator;
};

#define MSCCLPP_UNIQUE_ID_BYTES 128
struct UniqueId {
  char internal[MSCCLPP_UNIQUE_ID_BYTES];
};

/* Create a unique ID for communication. Only needs to be called by one process.
 * Use with mscclppCommInitRankFromId().
 * All processes need to provide the same ID to mscclppCommInitRankFromId().
 *
 * Outputs:
 *  uniqueId: the unique ID to be created
 */
std::unique_ptr<UniqueId> getUniqueId();

/* Transport Types */
enum class TransportType : uint8_t {
  P2P = 0,
  IB = 1,
};

class Communicator {
public:
  Communicator();
  ~Communicator();

  /* Initialize the communicator. nranks processes with rank 0 to nranks-1 need to call this function.
  *
  * Inputs:
  *   nranks:     number of ranks in the communicator
  *   ipPortPair: a string of the form "ip:port" that represents the address of the root process
  *   rank:       rank of the calling process
  */
  void initRank(int nranks, const char* ipPortPair, int rank);
  
  /* Initialize the communicator from a given UniqueId. Same as mscclppCommInitRank() except that
  * id is provided by the user by calling getUniqueId()
  *
  * Inputs:
  *   nranks: number of ranks in the communicator
  *   id:     the unique ID to be used for communication
  *   rank:   rank of the calling process
  */
  void initRankFromId(int nranks, UniqueId id, int rank);
  
  /* Ring-based AllGather through the bootstrap socket.
  *
  * Inputs:
  *   data: data array to be gathered where `[r*size, (r+1)*size)` is the data for rank `r`
  *   size: data size per rank
  */
  void bootstrapAllGather(void* data, int size);

  /* A no-op function that is used to synchronize all processes via a bootstrap allgather*/
  void bootstrapBarrier();

  /* Connect to a remote rank. This function only prepares metadata for connection. The actual connection
  * is made by a following call of mscclppConnectionSetup(). Note that this function is two-way and a connection
  * from rank i to remote rank j needs to have a counterpart from rank j to rank i.
  * Note that with IB, buffers are registered at a page level and if a buffer is spread through multiple pages
  * and do not fully utilize all of them, IB's QP has to register for all involved pages. This potentially has
  * security risks if the devConn's accesses are given to a malicious process.
  *
  * Inputs:
  *   remoteRank:    the rank of the remote process
  *   tag:           the tag of the connection. tag is copied into the corresponding mscclppDevConn_t, which can be
  *                  used to identify the connection inside a GPU kernel.
  *   transportType: the type of transport to be used (mscclppTransportP2P or mscclppTransportIB)
  *   ibDev:         the name of the IB device to be used. Expects a null for mscclppTransportP2P.
  */
  std::shared_ptr<HostConnection> connect(int remoteRank, int tag, TransportType transportType, const char* ibDev = 0);

  /* Establish all connections created by mscclppConnect(). This function must be called after all mscclppConnect()
  * calls are made. This function ensures that all remote ranks are ready to communicate when it returns.
  */
  void connectionSetup();
  
  /* Launch proxy thread(s). This function is supposed to be called before starting a kernel that uses DeviceConnection. */
  void startProxying();

  /* Stop proxy thread(s). */
  void stopProxying();
  
  /* Return the rank of the calling process.
  *
  * Outputs:
  *   rank: the rank of the calling process
  */
  int rank();

  /* Return the number of ranks of the communicator.
  *
  * Outputs:
  *   size: the number of ranks of the communicator
  */
  int size();

  struct Impl;
private:
  std::unique_ptr<Impl> pimpl;
  friend class HostConnection;
};

enum class ProxyHandlerResult {
  Continue,
  FlushAndContinue,
  Stop,
};

class Proxy;
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

class Proxy {
public:
  Proxy(ProxyHandler handler);

  ~Proxy();

  void start();

  void stop();
  
  HostProxyFifo& fifo();

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

} // namespace mscclpp

#endif // MSCCLPP_H_
