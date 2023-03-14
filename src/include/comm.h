/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_COMM_H_
#define MSCCLPP_COMM_H_

// #include "transport.h"
// #include "p2p.h"
// #include "collectives.h"
#include "proxy.h"
// #include "strongstream.h"
#include "ib.h"

// #if CUDART_VERSION < 9000
// struct cudaLaunchParams {
//   void *func;
//   dim3 gridDim;
//   dim3 blockDim;
//   void **args;
//   size_t sharedMem;
//   cudaStream_t stream;
// };
// #endif

// #define CACHE_LINE_SIZE 128
// #define MEM_ALIGN 4096
// #define CUDA_IPC_MIN 2097152UL

// // Channels / LL tuning
// #define MSCCLPP_LL_THREAD_THRESHOLD 8
// #define MSCCLPP_LL128_THREAD_THRESHOLD 8
// #define MSCCLPP_SIMPLE_THREAD_THRESHOLD 64

#define MAXCONNECTIONS 1024

// struct mscclppSendMem {
//   union {
//     struct {
//       uint64_t head;
//       char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
//       void* ptrExchange;
//       uint64_t redOpArgExchange[2];
//       char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
//       int offsFifo[MSCCLPP_STEPS];
//     };
//     char pad3[MEM_ALIGN];
//   };
// };

// struct mscclppRecvMem {
//   union {
//     struct {
//       uint64_t tail;
//       char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
//       int sizesFifo[MSCCLPP_STEPS];
//       int offsFifo[MSCCLPP_STEPS];
//       int flush; // For GDRCopy-based flush
//     };
//     char pad4[MEM_ALIGN];
//   };
// };

// enum helperThreadState {ThreadStart, ThreadStop};

// #define MSCCLPP_IPC_POOL_SIZE (2*MSCCLPP_MAX_LOCAL_RANKS*MSCCLPP_MAX_OPS)

// struct mscclppGraphHelperResources {
//   mscclppComm* comm;
//   pthread_mutex_t threadLock;
//   pthread_cond_t  threadCond;
//   enum helperThreadState threadState;
//   void* ipcBases[MSCCLPP_IPC_POOL_SIZE];
//   int ipcTail;
//   int ipcHead;
// };

// struct mscclppUserRedOp {
//   int freeNext; // -1=allocated, otherwise index of next free entry in array
//   mscclppDataType_t datatype;
//   mscclppDevRedOpFull opFull;
// };

// struct mscclppNodeRanks {
//   int localRanks;
//   int* localRankToRank;
// };

// struct mscclppDestructor {
//   struct mscclppDestructor* next;
//   void* obj;
//   mscclppResult_t(*fn)(struct mscclppDestructor* me);
// };

// struct mscclppCommCallback {
//   struct mscclppCommCallback* next;
//   mscclppResult_t(*fn)(struct mscclppComm* comm, struct mscclppCommCallback* cb);
// };

// struct mscclppChannel {
//   struct mscclppChannelPeer* peers;
//   struct mscclppDevChannelPeer* devPeers;
//   struct mscclppRing ring;
//   int* devRingUserRanks;
//   struct mscclppTree tree;
//   struct mscclppTree collnetChain;
//   struct mscclppDirect collnetDirect;
//   int id; // index of this channel
//   uint32_t workFifoSent; // last used work index+1
//   uint64_t p2pOpCount;
// };

// struct mscclppWorkList {
//   struct mscclppWorkList* next;
//   struct mscclppWork work;
// };

// struct mscclppPointerList {
//   struct mscclppPointerList* next;
//   void *ptr;
// };

// struct mscclppKernelPlan {
//   // A kernel plan is also a callback that reclaims itself. Hence this must
//   // be the first member.
//   struct mscclppCommCallback reclaimer;
//   struct mscclppMemoryPool memPool_mscclppProxyOp; // memory to return to comm in cleanup

//   struct mscclppComm* comm;
//   struct mscclppKernelPlan* next;

//   bool persistent; // aka captured in a graph
//   bool kernelSpecialized;
//   void *kernelFn;
//   int channelUbound; // only channels c < channelUbound are present
//   int channelCount; // number of channels present
//   uint64_t channelMask; // which channels are present, channelCount == popcount(channelMask)
//   bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
//   int threadPerBlock;
//   // workHeap fields are null until uploadWorkFifo() or preparePersistentKernel()
//   struct mscclppWork* workHead;

//   int collOpCount; // zero based for this plan

//   struct mscclppIntruQueue<struct mscclppPointerList, &mscclppPointerList::next> ipcMemQueue;

//   struct Channel {
//     int nWork;
//     union {
//       int nWorkElem; // used for coll and reg coll
//       int p2pTailElem[2]; // used for p2p, indexed by mscclppWorkElemP2pType-1
//     };
//     size_t collBytes;
//     struct mscclppIntruQueue<struct mscclppWorkList, &mscclppWorkList::next> workQueue;
//     struct mscclppIntruQueue<struct mscclppProxyOp, &mscclppProxyOp::enqNext> proxyOpQueue;
//   } channels[MAXCHANNELS];
// };

struct mscclppConn {
  mscclppTransport_t transport;
  int remoteRank;
  int buffSize;
  uint64_t *remoteProxyFlag;
  uint64_t *cpuProxyFlag;
  void *cpuProxyFlagGdrDesc;
  struct mscclppDevConn *devConn;
  struct mscclppIbContext *ibCtx;
  struct mscclppIbQp *ibQp;
  struct mscclppIbMr *ibBuffMr;
  struct mscclppIbMr *ibLocalFlagMr;
  struct mscclppIbMr *ibProxyFlagMr;
  struct mscclppIbMrInfo ibBuffMrInfo;
  struct mscclppIbMrInfo ibLocalFlagMrInfo;
  struct mscclppIbMrInfo ibProxyFlagMrInfo;
};

struct mscclppComm {
  struct mscclppConn conns[MAXCONNECTIONS];
  int nConns;

  void* bootstrap;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  // Flag to ask MSCCLPP kernels to abort
  volatile uint32_t *abortFlag;

  struct mscclppIbContext *ibContext[MSCCLPP_IB_MAX_DEVS];
  // Last one is for P2P proxies.
  struct mscclppProxyState proxyState[MSCCLPP_IB_MAX_DEVS + 1];
};

#endif
