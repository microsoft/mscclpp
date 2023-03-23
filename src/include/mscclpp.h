#ifndef MSCCLPP_H_
#define MSCCLPP_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <stdint.h>

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1
#define MSCCLPP_PROXY_FIFO_SIZE 8

#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 100 + MSCCLPP_MINOR)

#include <mscclppfifo.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************************************************
 * A mscclppDevConn provides a zero-copy connection between a sender and a receiver that are
 * connected via P2P NVLink or InfiniBand.
 * The communication API is one-sided meaning that for every single data transfer, only one side
 * needs to execute unlike a two-sided communication stack such as NCCL where both sides 
 * need to execute a send and a receive instruction respectively for every transfer.
 ***************************************************************************************************************
 * At connection setup time, a sender and the matching receiver need to call mscclppConnect to register 
 * their buffers locally. Once all buffers are registered via mscclppConnect, mscclppConnectionSetup is 
 * called to setup a bidirectional connection. With every connection, there is an associated CPU 
 * proxy thread that performs the actual data transfer using (R)DMA. DMA is optional for P2P NVLink connections
 * where the GPU can perform the copy directly.
 ***************************************************************************************************************
 * Before using any of functionality of connections, mscclppProxyLaunch needs to be called to spawn the 
 * proxy threads. There are currently two types of connections:
 * 
 * P2P via NVLink: the DMA engine can perform the copy between the buffers. DMA engine has higher latency
 * but has a higher bandwidth and costs no compute cycles on the GPU.
 * 
 * InfiniBand: the RDMA engine copies the data over MLX devices.
 ***************************************************************************************************************
 * At the runtime, a GPU kernel has access to a mscclppDevConn object that provides the following functions:
 * 
 * put(): the sender initiates a data transfer to the receiver.  
 * 
 * signal(): the sender signals the receiver that data is ready to be consumed once the reciver has performed a wait().
 * 
 * wait(): the reciever waits on the signal() to start reading the data.
 * 
 * The sender should not reuse the buffer till the signal returns.
 * The receiver should only access the data after the wait returns. 
 *   
 * putWithSignal(): The sender initiates a data transfer and signals the receiver that data is ready to be consumed. 
 * This is an optimized version of a put followed by a signal.
 * 
 * Example:
 * 
 * // sender GPU
 * devConn.put(data1)
 * devConn.put(data2)
 * devConn.put(data3)                                // receiver GPU
 * // not OK to write to data1, data2, data3         // not OK to read data1, data2, data3   
 * devConn.signal() -------------------------------> devConn.wait()
 * // OK to write to data1, data2, data3             // OK to read data1, data2, data3
 **************************************************************************************************************/
struct mscclppDevConn {
#ifdef __CUDACC__
 __forceinline__ __device__ void put(uint64_t dataOffset, uint64_t dataSize){
    fifo.push(mscclppData, dataOffset, dataSize);
  }

  __forceinline__ __device__ void signal(){
    epochIncrement();
    uint64_t curFifoHead = fifo.push(mscclppFlag | mscclppSync, 1, 1);
    while (*(volatile uint64_t *)fifo.triggerFifoTail <= curFifoHead);
  }

  __forceinline__ __device__ void putWithSignal(uint64_t dataOffset, uint64_t dataSize){
    epochIncrement();
    uint64_t curFifoHead = fifo.push(mscclppData | mscclppFlag | mscclppSync, dataOffset, dataSize);
    while (*(volatile uint64_t *)fifo.triggerFifoTail <= curFifoHead);
  }

  __forceinline__ __device__ void wait(){
    (*recvEpochId) += 1;
    // printf("%llu %llu %llu\n", (*(volatile uint64_t*)proxyEpochId), *(volatile uint64_t*)sendEpochId, *recvEpochId);
    while (*(volatile uint64_t*)proxyEpochId < (*recvEpochId));
  }

  __forceinline__ __device__ void epochIncrement(){
    *(volatile uint64_t*)sendEpochId += 1;
  }

#endif
  int tag;

  void* localBuff;
  uint64_t* sendEpochId;  // this is read and written by the GPU
  uint64_t* recvEpochId;   // this is the copy of the remote epoch id.

  void* remoteBuff;
  uint64_t* remoteFlag;
  uint64_t* proxyEpochId; // this is only written by the proxy thread

  // threads can access the fifo concurrently
  struct mscclppConcurrentFifo fifo;
};

typedef struct mscclppComm* mscclppComm_t;
typedef struct mscclppDevConn mscclppDevConn_t;

#define MSCCLPP_UNIQUE_ID_BYTES 128
typedef struct { char internal[MSCCLPP_UNIQUE_ID_BYTES]; } mscclppUniqueId;

/* Error type */
typedef enum { mscclppSuccess                 =  0,
               mscclppUnhandledCudaError      =  1,
               mscclppSystemError             =  2,
               mscclppInternalError           =  3,
               mscclppInvalidArgument         =  4,
               mscclppInvalidUsage            =  5,
               mscclppRemoteError             =  6,
               mscclppInProgress              =  7,
               mscclppNumResults              =  8 } mscclppResult_t;

mscclppResult_t mscclppGetUniqueId(mscclppUniqueId* uniqueId);

/* Transport Types */
typedef enum { mscclppTransportP2P = 0,
               mscclppTransportSHM = 1, // TODO(chhwang): not implemented yet
               mscclppTransportIB = 2,
} mscclppTransport_t;

mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, int rank, const char* ip_port_pair);

mscclppResult_t mscclppBootStrapAllGather(mscclppComm_t comm, void* data, int size);

mscclppResult_t mscclppCommDestroy(mscclppComm_t comm);

mscclppResult_t mscclppConnect(mscclppComm_t comm, int remoteRank, void* localBuff, size_t buffSize,
                               int tag, mscclppTransport_t transportType, const char *ibDev=NULL);

mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm);

mscclppResult_t mscclppGetDeviceConnections(mscclppComm_t comm, mscclppDevConn_t** devConns, int* nCons);

mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm);

mscclppResult_t mscclppProxyStop(mscclppComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MSCCLPP_H_
