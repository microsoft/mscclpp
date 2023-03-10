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
  mscclppTrigger *cpuTriggerFifo;
  // fifoTail indicates where CPU needs to read the head of the fifo. only accessible by CPU
  // No atomicity is required for fifoTail as only a single CPU thread accesses it.
  int fifoTail; 
  uint64_t *remoteProxyFlag;
  uint64_t *cpuProxyFlag;
  void *cpuTriggerFifoGdrDesc;
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
//   struct mscclppMemoryStack memPermanent, memScoped;
//   // List of destructors to run when comm is destructed
//   struct mscclppDestructor* destructorHead;

//   struct mscclppChannel channels[MAXCHANNELS];
//   struct mscclppPeerInfo* peerInfo;
//   struct mscclppTopoSystem* topo;

  struct mscclppConn conns[MAXCONNECTIONS];
  int nConns;

//   mscclppNet_t* mscclppNet;
//   mscclppCollNet_t* mscclppCollNet;
  void* bootstrap;
//   // Bitmasks for mscclppTransportP2pSetup
//   uint64_t* connectSend;
//   uint64_t* connectRecv;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
//   int compCap; // compute capability of the GPU
//   int64_t busId;   // my PCI bus ID in int format
//   cpu_set_t cpuAffinity; // CPU affinity of the GPU

  // int node;
  // int nNodes;
  // int localRank;
  // int localRanks;
  // int maxLocalRanks;
  // int* rankToNode;
  // int* rankToLocalRank;
  // int* localRankToRank; 
//   // localRanks and localRanktoRank for all nodes
//   struct mscclppNodeRanks* nodeRanks;

//   bool checkPointers;
//   bool dmaBufSupport;

//   // Counter for tracking CUDA launches (P2P and collectives included)
//   uint64_t opCount;
//   // Collective operation counter
//   uint64_t collOpCount;

//   // Channels for collectives
//   int nChannels;
//   // Channels (per peer) for p2p
//   int p2pnChannels;
//   int p2pnChannelsPerPeer;
//   int p2pChannels[MAXCHANNELS];

//   // Should this comm allocate LL buffers for network P2P connections?
//   bool allocP2pNetLLBuffers;

//   // Buffer sizes
//   int buffSizes[MSCCLPP_NUM_PROTOCOLS];
//   int p2pChunkSize;

//   // Algorithm/Protocols thresholds
//   ssize_t threadThresholds[MSCCLPP_NUM_ALGORITHMS][MSCCLPP_NUM_PROTOCOLS];
//   float latencies[MSCCLPP_NUM_FUNCTIONS][MSCCLPP_NUM_ALGORITHMS][MSCCLPP_NUM_PROTOCOLS];
//   float bandwidths[MSCCLPP_NUM_FUNCTIONS][MSCCLPP_NUM_ALGORITHMS][MSCCLPP_NUM_PROTOCOLS];
//   int maxThreads[MSCCLPP_NUM_ALGORITHMS][MSCCLPP_NUM_PROTOCOLS];

//   /* This attribute can indicate the states of communicators and return code of
//    * asynchronous MSCCLPP operations. */
//   mscclppResult_t asyncResult;

  // Flag to ask MSCCLPP kernels to abort
  volatile uint32_t *abortFlag;

//   // Device side of the communicator (for cudaFree's)
//   struct mscclppDevComm* devComm; // actually = &mscclppDevCommAndChannels::comm

//   // Operation pool.
//   int workFifoDepth; // size of workFifoHeap[], power of 2
//   struct mscclppWork* workFifoHeap;
//   struct mscclppWork* devWorkFifoHeap;
//   void* workFifoHeapGdrHandle;

//   // Work completion notificaion
//   uint32_t* workFifoDone/*[MAXCHANNELS]*/; // in cudaHost memory
//   uint32_t workFifoSent; // Monotonic (mod 1<<32) index of next unused fifo slot.
//   uint32_t workFifoAckdMin; // Monotonic index of least unprocessed fifo slot over all channels.

//   // Intra-process sync
//   struct mscclppComm* intraComm0; // leader of intra-process comms (self possible)
//   struct mscclppComm* intraNext; // next of intra-process comms, intraComm0 is head
//   int intraRank;
//   int intraRanks;
//   uint32_t intraBarrierPhase;
//   char intraPad1[64 - sizeof(uint64_t)];
//   uint64_t intraBarrierCounter; // only used if this is intraComm0
//   char intraPad2[64 - sizeof(uint64_t)];
//   uint64_t intraBarrierGate; // only used if this is intraComm0

  struct mscclppIbContext *ibContext[MSCCLPP_IB_MAX_DEVS];

  // Last one is for P2P proxies.
  struct mscclppProxyState proxyState[MSCCLPP_IB_MAX_DEVS + 1];

//   // Whether this communicator uses collNet
//   int collNetSupport;
//   int intraHighestTransportType;

//   size_t channelSize; // User requested work size (bytes) for channel partitions

//   // Internal streams
//   struct mscclppStrongStream deviceStream, hostStream;

//   // pools backed by comm->memPermanent
//   struct mscclppMemoryPool memPool_mscclppProxyOp;
//   struct mscclppMemoryPool memPool_mscclppKernelPlan;
//   struct mscclppMemoryPool memPool_mscclppPointerList;
//   // Next comm in this thread's active mscclppGroup[Start|End](). Holds "0x1" when
//   // this comm is not yet in a group.
//   struct mscclppComm* groupNext;
//   // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
//   struct mscclppComm* preconnectNext;
//   int persistentRefs; // number of persistent plan-lists capturing this comm
//   struct mscclppTasks tasks;

//   // user-created reduction ops
//   int userRedOpCapacity, userRedOpFreeHead;
//   mscclppUserRedOp *userRedOps;

//   // Queue of things for the main thread to do
//   struct mscclppIntruQueueMpsc<struct mscclppCommCallback, &mscclppCommCallback::next> callbackQueue;

//   // List of kernel plans built form tasks.
//   struct mscclppIntruQueue<struct mscclppKernelPlan, &mscclppKernelPlan::next> planQueue;
//   // First of the unlaunched kernels in `planQueue`
//   struct mscclppKernelPlan* unlaunchedPlansHead;

//   // communicator mode
//   int blocking;
//   // initState is to more conveniently reclaim resources when errors happen.
//   mscclppResult_t initState;
//   // flag to indicate if mscclppCommFinalize() is called
//   bool finalizeCalled;
//   // shared structures for finalization
//   int finalizeRankCnt;
};

// enum mscclppLaunchMode {
//   mscclppLaunchModeInvalid=0,
//   mscclppLaunchModeParallel,
//   mscclppLaunchModeGroup
// };
// extern enum mscclppLaunchMode mscclppParamLaunchMode;

// void mscclppCommPushFree(struct mscclppComm* comm, void* buf);
// void mscclppCommPushCudaFree(struct mscclppComm* comm, void* buf);
// void mscclppCommPushCudaHostFree(struct mscclppComm* comm, void* buf);
// void mscclppCommPushCudaGdrFree(struct mscclppComm* comm, void* handle);

// inline mscclppResult_t mscclppCommPollCallbacks(struct mscclppComm* comm, bool waitSome) {
//   mscclppResult_t result = mscclppSuccess;
//   struct mscclppCommCallback* cb = mscclppIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
//   while (cb != nullptr) {
//     struct mscclppCommCallback* next = cb->next;
//     mscclppResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
//     if (res1 != mscclppSuccess) result = res1;
//     cb = next;
//   }
//   MSCCLPPCHECK(result);
//   return mscclppSuccess;
// }

// inline void mscclppCommIntraBarrierIn(struct mscclppComm* comm, uint32_t x) {
//   int phase = comm->intraBarrierPhase;
//   if (comm->intraRanks == 1) {
//     // Release everyone (just me).
//     comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
//   } else {
//     struct mscclppComm* comm0 = comm->intraComm0;
//     uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
//     if (uint32_t(count) == uint32_t(comm->intraRanks)) {
//       // Reset.
//       __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
//       // Release everyone.
//       __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
//     }
//   }
// }

// // returns sum of x values contributed to mscclppCommIntraBarrierIn(comm, x)
// inline uint32_t mscclppCommIntraBarrierOut(struct mscclppComm* comm) {
//   struct mscclppComm* comm0 = comm->intraComm0;
//   comm->intraBarrierPhase ^= 1;
//   uint32_t phase = comm->intraBarrierPhase;
//   uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
//   if ((gate & 1) != phase) {
//     uint64_t t0 = clockNano();
//     do {
//       // Spin vigorously for first 5us.
//       if (clockNano()-t0 >= 5*1000) sched_yield();
//       gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
//     } while ((gate & 1) != phase);
//   }
//   if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
//   return gate>>32;
// }

// // Scrambles the bits of non-builtin values of mscclppRedOp_t according to the
// // communicator memory address. Used to catch bugs so that integer handles
// // associated with this communicator won't collide with handles of other
// // communicatrs. This function is its own inverse.
// static inline mscclppRedOp_t mscclppUserRedOpMangle(mscclppComm *comm, mscclppRedOp_t op) {
//   // Preserve the built-in values.
//   if(int(op) < int(mscclppNumOps))
//     return op;
//   uint64_t h = reinterpret_cast<uint64_t>(comm);
//   h ^= h >> 32;
//   h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
//   h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
//   h &= int(mscclppMaxRedOp); // mscclppMaxRedOp is a power of 2 minus 1
//   int op1 = int(h) ^ int(op);
//   // Since builtin values are preserved, we also have to preserve their preimage.
//   return op1 < int(mscclppNumOps) ? op : mscclppRedOp_t(op1);
// }

// mscclppResult_t mscclppCommEnsureReady(mscclppComm_t comm);
// mscclppResult_t mscclppCommSetAsyncError(mscclppComm_t comm, mscclppResult_t nextState);

#endif
