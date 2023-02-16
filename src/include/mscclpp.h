#ifndef MSCCLPP_H_
#define MSCCLPP_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1

#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 100 + MSCCLPP_MINOR)

#ifdef __cplusplus
extern "C" {
#endif

struct mscclppDevConn {
  int tag;

  void* localBuff;
  int* localFlag;
  // // remoteFlag <- localFlag
  // virtual void pushLocalFlag();
  // // remoteBuff[dstOffset..dstOffset+size-1] <- localBuff[srcOffset..srcOffset+size-1]
  // virtual void pushLocalBuff(size_t srcOffset, size_t dstOffset, size_t size);

  void* remoteBuff;
  int* remoteFlag;
  // // localFlag <- remoteFlag
  // virtual void pullRmoteFlag();
  // // localBuff[srcOffset..srcOffset+size-1] <- remoteBuff[dstOffset..dstOffset+size-1]
  // virtual void pullRemoteBuff(size_t srcOffset, size_t dstOffset, size_t size);
};

struct mscclppConn {
  mscclppTransport_t transport;
  // int localRank;
  int remoteRank;
  const char* ibDev;
  // int tag;
  // void* buff;
  // int* flag;
  mscclppDevConn* devConn;
};


typedef struct mscclppComm* mscclppComm_t;
typedef struct mscclppDevConn* mscclppDevConn_t;

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
               mscclppTransportSHM = 1,
               mscclppTransportIB = 2,
} mscclppTransport_t;

mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, int rank, const char* ip_port_pair);

mscclppResult_t mscclppBootStrapAllGather(mscclppComm_t comm, void* data, int size);

mscclppResult_t mscclppCommDestroy(mscclppComm_t comm);

mscclppResult_t mscclppConnect(mscclppComm_t comm, mscclppDevConn* devConnOut, int remoteRank, void* localBuff, int* localFlag, int tag,
                               mscclppTransport_t transportType, const char *ibDev=NULL);

mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm);

mscclppResult_t mscclppGetDevConns(mscclppComm_t comm, mscclppDevConn_t* devConns);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MSCCLPP_H_
