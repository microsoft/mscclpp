#ifndef MSCCLPP_H_
#define MSCCLPP_H_

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1
#define MSCCLPP_PATCH 0
#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 10000 + MSCCLPP_MINOR * 100 + MSCCLPP_PATCH)

// For every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER, a flush of the tail to device memory is triggered.
// As long as MSCCLPP_PROXY_FIFO_SIZE is large enough, having a stale tail is not a problem.
#define MSCCLPP_PROXY_FIFO_SIZE 128
#define MSCCLPP_PROXY_FIFO_FLUSH_COUNTER 4

#include <mscclppfifo.h>
#include <time.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct alignas(16) mscclppDevConnSignalEpochId
{
  // every signal(), increaments this and either:
  // 1) proxy thread pushes it to the remote peer's localSignalEpochId->proxy
  // 2) gpu thread directly writes it to remoteSignalEpochId->device
  uint64_t device;
  // signal() function triggers the cpu proxy thread to write to it
  uint64_t proxy;
};

struct mscclppBaseConn
{
  int remoteRank;
  int tag;

  // my local buffer
  void* localBuff;

  struct mscclppDevConnSignalEpochId* localSignalEpochId;
  // used by the signal() function directly from gpu
  struct mscclppDevConnSignalEpochId* remoteSignalEpochId;

  // every wait(), increaments this and then the gpu waits for either:
  // 1) localSignalEpochId->proxy to be >= this in case of a proxy thread
  // 2) remoteSignalEpochId->device to be >= this in case of a gpu thread
  uint64_t* waitEpochId;

  // my remote peer's buffer. only non-NULL with gpu's direct access
  // gpu can directly write into it
  void* remoteBuff;
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
struct mscclppDevConn : mscclppBaseConn
{
#ifdef __CUDACC__
  __forceinline__ __device__ void put(uint64_t dstDataOffset, uint64_t srcDataOffset, uint64_t dataSize)
  {
    fifo.push(mscclppData, dstDataOffset, srcDataOffset, dataSize);
  }

  __forceinline__ __device__ void put(uint64_t dataOffset, uint64_t dataSize)
  {
    put(dataOffset, dataOffset, dataSize);
  }

  __forceinline__ __device__ void signal()
  {
    epochIncrement();
    fifo.push(mscclppFlag, 0, 0, 1);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t dstDataOffset, uint64_t srcDataOffset, uint64_t dataSize)
  {
    epochIncrement();
    fifo.push(mscclppData | mscclppFlag, dstDataOffset, srcDataOffset, dataSize);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t dataOffset, uint64_t dataSize)
  {
    putWithSignal(dataOffset, dataOffset, dataSize);
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstDataOffset, uint64_t srcDataOffset,
                                                        uint64_t dataSize)
  {
    epochIncrement();
    uint64_t curFifoHead = fifo.push(mscclppData | mscclppFlag | mscclppSync, dstDataOffset, srcDataOffset, dataSize);
    while (*(volatile uint64_t*)&fifo.triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo.triggerFifoTail <= curFifoHead)
      ;
  }

  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dataOffset, uint64_t dataSize)
  {
    putWithSignalAndFlush(dataOffset, dataOffset, dataSize);
  }

  __forceinline__ __device__ void flush()
  {
    uint64_t curFifoHead = fifo.push(mscclppSync, 0, 0, 1);
    // we need to wait for two conditions to be met to ensure the CPU is done flushing. (1) wait for the tail
    // to go pass by curFifoHead (this is safety net) and (2) wait for the work element value to change to 0.
    while (*(volatile uint64_t*)&fifo.triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0 &&
           *(volatile uint64_t*)fifo.triggerFifoTail <= curFifoHead)
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

#endif
  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  struct mscclppConcurrentFifo fifo;
};

typedef struct mscclppComm* mscclppComm_t;
typedef struct mscclppDevConn mscclppDevConn_t;

#define MSCCLPP_UNIQUE_ID_BYTES 128
typedef struct
{
  char internal[MSCCLPP_UNIQUE_ID_BYTES];
} mscclppUniqueId;

// MR info to be shared with the remote peer
struct mscclppIbMrInfo
{
  uint64_t addr;
  uint32_t rkey;
};

// IB memory region
struct mscclppIbMr
{
  struct ibv_mr* mr;
  void* buff;
  struct mscclppIbMrInfo info;
};

struct mscclppRegisteredMemoryP2P
{
  void* remoteBuff;
  mscclppIbMr* IbMr;
};

struct mscclppRegisteredMemory
{
  std::vector<mscclppRegisteredMemoryP2P> p2p;
};

/* Error type */
typedef enum
{
  mscclppSuccess = 0,
  mscclppUnhandledCudaError = 1,
  mscclppSystemError = 2,
  mscclppInternalError = 3,
  mscclppInvalidArgument = 4,
  mscclppInvalidUsage = 5,
  mscclppRemoteError = 6,
  mscclppInProgress = 7,
  mscclppNumResults = 8
} mscclppResult_t;

/* Create a unique ID for communication. Only needs to be called by one process.
 * Use with mscclppCommInitRankFromId().
 * All processes need to provide the same ID to mscclppCommInitRankFromId().
 *
 * Outputs:
 *  uniqueId: the unique ID to be created
 */
mscclppResult_t mscclppGetUniqueId(mscclppUniqueId* uniqueId);

/* Transport Types */
typedef enum
{
  mscclppTransportP2P = 0,
  mscclppTransportSHM = 1, // TODO(chhwang): not implemented yet
  mscclppTransportIB = 2,
} mscclppTransport_t;

/* Initialize a communicator. nranks processes with rank 0 to nranks-1 need to call this function.
 *
 * Outputs:
 *   comm: the communicator to be initialized
 *
 * Inputs:
 *   nranks:     number of ranks in the communicator
 *   ipPortPair: a string of the form "ip:port" that represents the address of the root process
 *   rank:       rank of the calling process
 */
mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, const char* ipPortPair, int rank);

/* Initialize a communicator from a given mscclppUniqueId. Same as mscclppCommInitRank() except that
 * id is provided by the user by calling mscclppGetUniqueId()
 *
 * Outputs:
 *   comm: the communicator to be initialized
 *
 * Inputs:
 *   nranks: number of ranks in the communicator
 *   id:     the unique ID to be used for communication
 *   rank:   rank of the calling process
 */
mscclppResult_t mscclppCommInitRankFromId(mscclppComm_t* comm, int nranks, mscclppUniqueId id, int rank);

/* Ring-based AllGather through the bootstrap socket.
 *
 * Outputs:
 *   comm: the communicator
 *
 * Inputs:
 *   data: data array to be gathered where `[r*size, (r+1)*size)` is the data for rank `r`
 *   size: data size per rank
 */
mscclppResult_t mscclppBootstrapAllGather(mscclppComm_t comm, void* data, int size);

/* A no-op function that is used to synchronize all processes via a bootstrap allgather*/
mscclppResult_t mscclppBootstrapBarrier(mscclppComm_t comm);

/* Destroy a communicator.
 *
 * Inputs:
 *   comm: the communicator to be destroyed
 */
mscclppResult_t mscclppCommDestroy(mscclppComm_t comm);

/* Return the string for the given error code.
 *
 * Ouput:
 *   returns the string
 *
 * Inputs:
 *   result: the error code that this function needs to translate
 */
const char* mscclppGetErrorString(mscclppResult_t result);

/* Connect to a remote rank. This function only prepares metadata for connection. The actual connection
 * is made by a following call of mscclppConnectionSetup(). Note that this function is two-way and a connection
 * from rank i to remote rank j needs to have a counterpart from rank j to rank i.
 * Note that with IB, buffers are registered at a page level and if a buffer is spread through multiple pages
 * and do not fully utilize all of them, IB's QP has to register for all involved pages. This potentially has
 * security risks if the devConn's accesses are given to a malicious process.
 *
 * Inputs:
 *   comm:          the communicator
 *   remoteRank:    the rank of the remote process
 *   tag:           the tag of the connection. tag is copied into the corresponding mscclppDevConn_t, which can be
 *                  used to identify the connection inside a GPU kernel.
 *   localBuff:     the local send/receive buffer
 *   buffSize:      the size of the local buffer
 *   transportType: the type of transport to be used (mscclppTransportP2P or mscclppTransportIB)
 *   ibDev:         the name of the IB device to be used. Expects a null for mscclppTransportP2P.
 */
mscclppResult_t mscclppConnect(mscclppComm_t comm, int remoteRank, int tag, void* localBuff, uint64_t buffSize,
                               mscclppTransport_t transportType, const char* ibDev = 0);

/* Establish all connections declared by mscclppConnect(). This function must be called after all mscclppConnect()
 * calls are made. This function ensures that all remote ranks are ready to communicate when it returns.
 *
 * Inputs:
 *   comm: the communicator
 */
mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm);

/* Return an array of mscclppDevConn_t and the number of connections created by mscclppConnectionSetup().
 * The order of connections matches the order of mscclppConnect() calls.
 *
 * Outputs:
 *   devConns: the array of mscclppDevConn_t. Each mscclppDevConn_t corresponds to a mscclppConnect() call in the
 *             order of the calls.
 *   nConns:   the number of connections
 *
 * Inputs:
 *   comm: the communicator
 */
mscclppResult_t mscclppGetAllDeviceConnections(mscclppComm_t comm, mscclppDevConn_t** devConns, int* nConns);

/* Return the mscclppDevConn_t corresponding to a given tag and a remoteRank.
 *
 * Outputs:
 *   devConn: the mscclppDevConn_t corresponding to the given tag
 *
 * Inputs:
 *   comm:       the communicator
 *   tag:        the tag of the connection
 *   remoteRank: the remoteRank of the connection
 */
mscclppResult_t mscclppGetDeviceConnection(mscclppComm_t comm, int remoteRank, int tag, mscclppDevConn_t** devConn);

/* Launch proxy threads for all connections created by mscclppConnectionSetup(). This function is supposed to be called
 * before starting a kernel that uses mscclppDevConn_t. Up to two proxy threads are launched for each (GPU + IB) pair
 * (one for P2P NVLink and one for InfiniBand).
 *
 * Inputs:
 *  comm: the communicator
 */
mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm);

/* Stop all proxy threads.
 *
 * Inputs:
 *  comm: the communicator
 */
mscclppResult_t mscclppProxyStop(mscclppComm_t comm);

/* Return the rank of the calling process.
 *
 * Outputs:
 *   rank: the rank of the calling process
 *
 * Inputs:
 *   comm: the communicator
 */
mscclppResult_t mscclppCommRank(mscclppComm_t comm, int* rank);

/* Return the number of ranks of the communicator.
 *
 * Outputs:
 *   size: the number of ranks of the communicator
 *
 * Inputs:
 *   comm: the communicator
 */
mscclppResult_t mscclppCommSize(mscclppComm_t comm, int* size);

/* Set the timeout for the bootstrap connection.
 *
 * Inputs:
 *   timeout: the timeout in seconds
 */
mscclppResult_t mscclppSetBootstrapConnTimeout(int timeout);

/* Log handler type which is a callback function for
 * however user likes to handle the log messages. Once set,
 * the logger will just call this function with msg.
 */
typedef void (*mscclppLogHandler_t)(const char* msg);

/* The default log handler.
 *
 * Inputs:
 *   msg: the log message
 */
void mscclppDefaultLogHandler(const char* msg);

/* Set a custom log handler.
 *
 * Inputs:
 *   handler: the log handler function
 */
mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler);

/* Register a buffer for RDMA.
 *
 * Outputs:
 *   regMem: the registered memory
 *
 * Inputs:
 *   comm:        the communicator
 *   local_memory: the local buffer to be registered
 *   size:        the size of the buffer
 */
mscclppResult_t mscclppRegisterBuffer(mscclppComm_t comm, void* local_memory, size_t size,
                                      mscclppRegisteredMemory* regMem);

/* Write to a registered buffer.
 *
 * Inputs:
 *   comm:        the communicator
 *   regMem:      the registered memory
 *   srcBuff:     the source buffer
 *   size:        the size of the buffer
 *   srcOffset:   the offset of the source buffer
 *   dstOffset:   the offset of the destination buffer
 *   stream:      the CUDA stream
 */
mscclppResult_t mscclppRegisteredBufferWrite(mscclppComm_t comm, mscclppRegisteredMemory* regMem, void* srcBuff,
                                             size_t size, uint32_t srcOffset, uint32_t dstOffset, int64_t stream);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MSCCLPP_H_
