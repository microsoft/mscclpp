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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum : uint64_t { mscclppData = 0x1,
                          mscclppFlag = 0x2,
                          mscclppSync = 0x4} mscclppTriggerType_t;

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10

// the summation of number of bits must be 128 or less
union alignas(16) mscclppTrigger {
  uint64_t value[2];
  struct {
    // first 64 bits: value[0]
    uint64_t dataSize   : MSCCLPP_BITS_SIZE;
    uint64_t dataOffset : MSCCLPP_BITS_OFFSET;
    uint64_t            : (64-MSCCLPP_BITS_SIZE-MSCCLPP_BITS_OFFSET); // ensure 64-bit alignment
    // second 64 bits: value[1]
    uint64_t connId     : MSCCLPP_BITS_CONNID;
    uint64_t type       : MSCCLPP_BITS_TYPE;
    uint64_t            : (64-MSCCLPP_BITS_CONNID-MSCCLPP_BITS_TYPE); // ensure 64-bit alignment
  } fields;
};

typedef uint64_t mscclppRequest_t;
typedef mscclppTrigger* mscclppTrigger_t;

struct mscclppConcurrentFifo {
#ifdef __CUDACC__
  __forceinline__ __device__ mscclppRequest_t getTrigger(mscclppTrigger_t* trig) {
    uint64_t curFifoHead = atomicAdd((unsigned long long int*)this->triggerFifoHead,1);
    while (curFifoHead >= MSCCLPP_PROXY_FIFO_SIZE + *((volatile uint64_t*)this->triggerFifoTail));
    *trig = &this->triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE];
    return curFifoHead;
  }

  __forceinline__ __device__ void setTrigger(mscclppTrigger_t trig, uint64_t type, uint64_t dataOffset, uint64_t dataSize) {
    asm volatile(
      "st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(&trig->value),
      "l"((dataOffset << (MSCCLPP_BITS_SIZE)) +
          (dataSize)),
      "l"((type << MSCCLPP_BITS_CONNID) + this->connId));
  }

  __forceinline__ __device__ void waitTrigger(mscclppRequest_t req) {
    while (*(volatile uint64_t *)triggerFifoTail <= req);
  }
#endif // __CUDACC__
  mscclppTrigger* triggerFifo;
  uint64_t* triggerFifoTail; // read by both device and host. written only by host
  uint64_t* triggerFifoHead; // read by both device and host. written only by device
  int connId;
};


/***************************************************************************************************************
 * A mscclppDevConn provides a zero-copy connection between a sender and a receiver that are
 * connected via P2P NVLink or IB.
 * The communication API is one-sided meaning that not both side of a connection are involved
 * in a single transfer. This is unlike NCCL/MSCCL where for each send instruction, there needs 
 * to be a matching receive instruction. MPI_Put and MPI_Get are the closest programming model 
 * in MSCCL++.
 * 
 * At connection setup, the sender and receiver register the respective buffers through mscclppConnect.
 * 
 * After connection setup, if the connection type is:
 *    P2P via NVLink: mscclppDevConn has access to remoteBuff and remoteFlag
 *    InfiniBand: mscclppDevConn has no access to remoteBuff or remoteFlag
 * 
 * For any connection, there is a proxy thread associated with it:
 *    P2P via NVLink: the DMA engine can perform the copy between the buffers. DMA engine has higher latency
 *    but has a higher bandwidth and costs no compute cycles on the GPU.
 *    InfiniBand: the RDMA engine copies the data over via MLX devices.
 * 
 * Memory consistency:
 *    In general, there is no guarantee on the order in which bytes are received. MSCCL++ relies on the following
 *    property to meet memory consistency: consecutive wirtes/reads by the CPU proxy are observed by the GPU in the same order
 *    as they are issued in. This means that for a sequence of writes done by a CPU proxy, we need to write a synchornization
 *    value written in flag that the receiving side of the GPU needs to poll on to ensure the arrival of writes.
 *
 * The communication from GPU to CPU proxy happens via trigger which is allocated on the GPU global memory and mounted on the CPU
 * with GDR copy. The CPU proxy has a fifo of work elements which are communicated via trigger. getTrigger gets a place on the fifo
 * (note that an atomicInc is used to enable concurrent calls to getTrigger). setTrigger rights the right work element to the fifo
 * so that the CPU proxy can consume it.
 * 
 **************************************************************************************************************/
struct mscclppDevConn {
  int tag;

  void* localBuff;
  uint64_t* localFlag;

  void* remoteBuff;
  uint64_t* remoteFlag;
  uint64_t* proxyFlag; // this is only written by the proxy thread

  // multiple threads can access the fifo concurrently
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

/* Reduction operation selector */
typedef enum { mscclppNumOps_dummy = 5 } mscclppRedOp_dummy_t;
typedef enum { mscclppSum        = 0,
               mscclppProd       = 1,
               mscclppMax        = 2,
               mscclppMin        = 3,
               mscclppAvg        = 4,
               /* mscclppNumOps: The number of built-in mscclppRedOp_t values. Also
                * serves as the least possible value for dynamic mscclppRedOp_t's
                * as constructed by mscclppRedOpCreate*** functions. */
               mscclppNumOps     = 5,
               /* mscclppMaxRedOp: The largest valid value for mscclppRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(mscclppRedOp_t) when compared to previous MSCCLPP versions to
                * maintain ABI compatibility. */
               mscclppMaxRedOp   = 0x7fffffff>>(32-8*sizeof(mscclppRedOp_dummy_t))
             } mscclppRedOp_t;

/* Data types */
typedef enum { mscclppInt8       = 0, mscclppChar       = 0,
               mscclppUint8      = 1,
               mscclppInt32      = 2, mscclppInt        = 2,
               mscclppUint32     = 3,
               mscclppInt64      = 4,
               mscclppUint64     = 5,
               mscclppFloat16    = 6, mscclppHalf       = 6,
               mscclppFloat32    = 7, mscclppFloat      = 7,
               mscclppFloat64    = 8, mscclppDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
               mscclppBfloat16   = 9,
               mscclppNumTypes   = 10
#else
               mscclppNumTypes   = 9
#endif
} mscclppDataType_t;

/* Transport Types */
typedef enum { mscclppTransportP2P = 0,
               mscclppTransportSHM = 1, // TODO(chhwang): not implemented yet
               mscclppTransportIB = 2,
} mscclppTransport_t;

mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, int rank, const char* ip_port_pair);

mscclppResult_t mscclppBootStrapAllGather(mscclppComm_t comm, void* data, int size);

mscclppResult_t mscclppCommDestroy(mscclppComm_t comm);

mscclppResult_t mscclppConnect(mscclppComm_t comm, mscclppDevConn* devConnOut, int remoteRank, void* localBuff, size_t buffSize,
                               uint64_t* localFlag, int tag, mscclppTransport_t transportType, const char *ibDev=NULL);

mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm);

mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm);

mscclppResult_t mscclppProxyStop(mscclppComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MSCCLPP_H_
