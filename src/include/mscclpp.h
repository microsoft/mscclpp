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

#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 100 + MSCCLPP_MINOR)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { mscclppData = 0x1,
               mscclppFlag = 0x2,
               mscclppSync = 0x4} mscclppTriggerType_t;

#define MSCCLPP_SIZE_BITS 30
#define MSCCLPP_OFFSET_BITS 31

#define TRIGGER_VALUE(__TYPE__,__OOFSET__,__SIZE__) (((((__TYPE__) << MSCCLPP_OFFSET_BITS) + (__OFFSET__)) << MSCCLPP_SIZE_BITS) + __SIZE__ )

// the summation of number of bits must be 64 or less
union alignas(8) mscclppTrigger {
  uint64_t value;
  struct {
    uint64_t dataSize : MSCCLPP_SIZE_BITS;
    uint64_t dataOffset : MSCCLPP_OFFSET_BITS;
    uint64_t type : 3;
  } fields;
};

/**************************************
 * A mscclppDevConn provides a zero-copy connection between a sender and a receiver.
 * It contains a send_buffer and a recv_buffer of the same size
 * 
 * At connection setup, the sender and receiver register the respective buffers through mscclppConnect.
 * 
 * After connection setup, 
 *    mscclppDevConn has exclusive ownership of the recv_buffer; 
 *    the sender has exclusive ownership of the send_buffer
 * 
 * The Push communication proceeds as follows:
 * 1. Sender calls mscclppDevConn::asyncSend() once the contents of the send_buffer are ready
 *    Now both the sender and mscclppDevConn have shared (read) ownership of the send_buffer
 *    mscclppDevConn synchronously waits for the exclusive ownership of the recv_buffer (for previous recv to finish),
 *    initiates the copy to the recv_buffer, and returns
 * 
 * 2. Sender calls mscclppDevConn::waitSend() to wait for the copy to complete.
 *    When this call returns, the sender has exclusive ownership of the send_buffer again; mscclppDevConn has no ownership
 * 
 * 3. Receiver calls mscclppDevConn::waitRecv() to wait for the copy to complete.
 *    When this call returns, the receiver has exclusive ownership of the recv_buffer; mscclppDevConn has no ownership
 * 
 * 4. Receiver calls mscclppDevConn::recvDone() to indicate that it is done with the recv_buffer
 * 
 ***************************************/

struct mscclppDevConn {
  int tag;

  void* localBuff;
  uint64_t* localFlag;
  // // remoteFlag <- localFlag
  // virtual void pushLocalFlag();
  // // remoteBuff[dstOffset..dstOffset+size-1] <- localBuff[srcOffset..srcOffset+size-1]
  // virtual void pushLocalBuff(size_t srcOffset, size_t dstOffset, size_t size);

  void* remoteBuff;
  uint64_t* remoteFlag;
  // // localFlag <- remoteFlag
  // virtual void pullRmoteFlag();
  // // localBuff[srcOffset..srcOffset+size-1] <- remoteBuff[dstOffset..dstOffset+size-1]
  // virtual void pullRemoteBuff(size_t srcOffset, size_t dstOffset, size_t size);

  mscclppTrigger* trigger;
  uint64_t* proxyFlag;
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
