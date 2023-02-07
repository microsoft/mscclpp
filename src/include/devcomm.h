/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_DEVICE_H_
#define MSCCLPP_DEVICE_H_

#include "mscclpp.h"
#include "align.h"
#include <stdint.h>

// #define MSCCLPP_NUM_FUNCTIONS 5 // Send/Recv not included for now
// typedef enum { mscclppFuncBroadcast, mscclppFuncReduce, mscclppFuncAllGather, mscclppFuncReduceScatter, mscclppFuncAllReduce, mscclppFuncSendRecv, mscclppFuncSend, mscclppFuncRecv, mscclppNumFuncs} mscclppFunc_t;
// extern const char* mscclppFuncStr[MSCCLPP_NUM_FUNCTIONS];

// #define MSCCLPP_NUM_ALGORITHMS 4 // Tree/Ring/CollNet*
// #define MSCCLPP_ALGO_TREE 0
// #define MSCCLPP_ALGO_RING 1
// #define MSCCLPP_ALGO_COLLNET_DIRECT 2
// #define MSCCLPP_ALGO_COLLNET_CHAIN 3
// extern const char* mscclppAlgoStr[MSCCLPP_NUM_ALGORITHMS];

#define MSCCLPP_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define MSCCLPP_PROTO_LL 0
#define MSCCLPP_PROTO_LL128 1
#define MSCCLPP_PROTO_SIMPLE 2
extern const char* mscclppProtoStr[MSCCLPP_NUM_PROTOCOLS];

// #define MSCCLPP_MAX_OPS 2048
// #define MSCCLPP_STEPS 8

// union mscclppLLFifoLine {
//   /* Flags have to be *after* data, because otherwise, an incomplete receive
//      from the network may receive the flag but not the data.
//      Note this is assuming that either we receive contiguous chunks of data
//      (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
//   struct {
//     uint32_t data1;
//     uint32_t flag1;
//     uint32_t data2;
//     uint32_t flag2;
//   };
//   uint64_t v[2];
//   int4 i4;
// };

// #define WARP_SIZE 32
#define MAXCHANNELS 32
// #define MSCCLPP_MAX_NTHREADS 640
// #define MSCCLPP_SIMPLE_MAX_NTHREADS 512
// #define MSCCLPP_LL_MAX_NTHREADS 512
// #define MSCCLPP_LL_LINES_PER_THREAD 8
// #ifdef TEST_LL_CLEANUP
// #define MSCCLPP_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
// #define MSCCLPP_LL_FLAG_MAX   0x100
// #define MSCCLPP_LL_FLAG(a) ((uint32_t)((a) % MSCCLPP_LL_FLAG_MAX))
// #else
// #define MSCCLPP_LL_CLEAN_MASK 0x7ffffff8
// #define MSCCLPP_LL_FLAG(a) ((uint32_t)(a))
// #endif
// // Make sure the clean mask will last for at least MSCCLPP_NSTEPS
// static_assert(MSCCLPP_LL_CLEAN_MASK % MSCCLPP_STEPS == 0, "Invalid MSCCLPP_LL_CLEAN_MASK value");

// #define MSCCLPP_LL128_LINESIZE 128
// #define MSCCLPP_LL128_LINEELEMS (MSCCLPP_LL128_LINESIZE/sizeof(uint64_t))
// #define MSCCLPP_LL128_DATAELEMS (MSCCLPP_LL128_LINEELEMS-1)

// #define MSCCLPP_LL128_MAX_NTHREADS 640
// #define MSCCLPP_LL128_ELEMS_PER_THREAD 120

// #define MSCCLPP_LL128_SHMEM_ELEMS_PER_THREAD 8
// #define MSCCLPP_LL128_SHMEM_SIZE (MSCCLPP_LL128_SHMEM_ELEMS_PER_THREAD*MSCCLPP_LL128_MAX_NTHREADS)

// #define MSCCLPP_DIRECT_WRITE 0x01
// #define MSCCLPP_DIRECT_READ  0x02
// #define MSCCLPP_DIRECT_NIC   0x04
// #define MSCCLPP_IPC_WRITE    0x08
// #define MSCCLPP_IPC_READ     0x10

struct mscclppConnInfo {
  // Regular comm mechanism
  char *buffs[MSCCLPP_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct mscclppProxyConnector {
  int rank;
  int localRank;
  struct mscclppProxyConnection* connection;
  struct mscclppComm* comm;
};

struct mscclppConnector {
  int connected;
  struct mscclppProxyConnector proxyConn;
  struct mscclppTransportComm* transportComm;
  void* transportResources;
  struct mscclppConnInfo conn;
  struct mscclppComm *comm;
};

struct mscclppRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal mscclpp index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};


#define MSCCLPP_MAX_TREE_ARITY 3
struct mscclppTree {
  int depth;
  int up;
  int down[MSCCLPP_MAX_TREE_ARITY];
};

#define MSCCLPP_MAX_DIRECT_ARITY 7
struct mscclppDirect {
  int depth;
  int out;
  int nHeads;
  int headRank;
  int shift;
  int up[MSCCLPP_MAX_DIRECT_ARITY];
  int down[MSCCLPP_MAX_DIRECT_ARITY];
};

#define MSCCLPP_MAX_CONNS 2
struct mscclppChannelPeer {
  struct mscclppConnector send[MSCCLPP_MAX_CONNS];
  struct mscclppConnector recv[MSCCLPP_MAX_CONNS];
};

// struct mscclppDevComm;

// /* mscclppWork is to be a power of two, currently 8x64 bytes, */
// /* to make sure reads to host from the CUDA kernel are aligned. */
// /* Make sure to adjust padding at the end of mscclppWorkElem. */
// #define MSCCLPP_WORK_SIZE 512

// enum mscclppWorkType : uint8_t {
//    mscclppWorkTypeUnused=0,
//    mscclppWorkTypeColl=1,
//    mscclppWorkTypeP2p=2,
//    mscclppWorkTypeRegColl=3
// };
// enum mscclppWorkP2PType : uint8_t {
//   mscclppWorkP2pTypeUnused=0,
//   mscclppWorkP2pTypeSend,
//   mscclppWorkP2pTypeRecv
// };

// struct mscclppWorkHeader {
//   union {
//     int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
//     uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send back.
//   };
//   uint16_t funcIndex;
//   uint8_t isLast:1; // last work for this kernel
//   uint8_t inFifo:1; // is this work in the fifo
//   enum mscclppWorkType type;
// };

// struct mscclppWorkElem {
//   union {
//     uint8_t flagBits;
//     struct {
//       uint8_t isUsed:1, redOpArgIsPtr:1, regUsed:1;
//     };
//   };
//   uint8_t nWarps;
//   uint8_t direct;

//   const void * sendbuff;
//   void * recvbuff;

//   size_t count;
//   size_t lastChunkSize;
//   uint32_t root;
//   uint8_t bid;
//   uint8_t nChannels;
//   uint64_t redOpArg;
// };

// #define MSCCLPP_MAX_WORK_ELEMENTS ((MSCCLPP_WORK_SIZE - alignUp(sizeof(mscclppWorkHeader), alignof(mscclppWorkElem)))/sizeof(mscclppWorkElem))
// static_assert(MSCCLPP_MAX_WORK_ELEMENTS == 9, "Sanity check: MSCCLPP_MAX_WORK_ELEMENTS == 9");

// struct mscclppWorkElemP2p {
//   int peer : 30;
//   int proto : 2;

//   enum mscclppWorkP2PType p2pType;
//   uint8_t nWarps;
//   uint8_t warpStart;
//   uint8_t ngroups;
//   // Important not to use any fields with greater than 4-byte alignment since
//   // we need sizeof(mscclppWorkElemP2p)==28, but that would be padded up to 32 if
//   // there were 8-byte fields.
//   //void* buff;
//   uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
//   //size_t count;
//   uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
//   int chunkSize;
// };

// static_assert(((MSCCLPP_WORK_SIZE - alignUp(sizeof(mscclppWorkHeader), alignof(mscclppWorkElemP2p)))/sizeof(mscclppWorkElemP2p)) >= 16, "Sanity check: MSCCLPP_MAX_WORK_ELEMENTS_P2P == 16");
// #define MSCCLPP_MAX_WORK_ELEMENTS_P2P 16

// struct mscclppWorkElemReg {
//   struct mscclppWorkElem elem;
//   void* dnInputs[MSCCLPP_MAX_DIRECT_ARITY+1];
//   void* dnOutputs[MSCCLPP_MAX_DIRECT_ARITY+1];
//   void* upOutputs[MSCCLPP_MAX_DIRECT_ARITY+1];
// };

// #define MSCCLPP_MAX_WORK_ELEMENTS_REG ((MSCCLPP_WORK_SIZE - alignUp(sizeof(mscclppWorkHeader), alignof(mscclppWorkElemReg)))/sizeof(mscclppWorkElemReg))
// static_assert(MSCCLPP_MAX_WORK_ELEMENTS_REG == 2, "Sanity check: MSCCLPP_MAX_WORK_ELEMENTS_REG == 2");

// // Number of named barriers supported by CUDA
// #define MSCCLPP_MAX_GROUPS 16

// struct mscclppWork {
//   struct mscclppWorkHeader header;
//   union {
//     char pad[MSCCLPP_WORK_SIZE - sizeof(struct mscclppWorkHeader)];
//     struct mscclppWorkElem elems[MSCCLPP_MAX_WORK_ELEMENTS];
//     struct mscclppWorkElemP2p p2pElems[MSCCLPP_MAX_WORK_ELEMENTS_P2P];
//     struct mscclppWorkElemReg regElems[MSCCLPP_MAX_WORK_ELEMENTS_REG];
//   };
// };
// static_assert(sizeof(struct mscclppWork) == MSCCLPP_WORK_SIZE, "Sanity check: sizeof(struct mscclppWork) == MSCCLPP_WORK_SIZE");
// static_assert(sizeof(struct mscclppWork)%16 == 0, "Sanity check: sizeof(struct mscclppWork)%16 == 0");

struct mscclppDevChannelPeer {
  // Stripped version of mscclppChannelPeer where we only keep the mscclppConnInfo
  // instead of the full mscclppConnector.
  struct mscclppConnInfo send[MSCCLPP_MAX_CONNS];
  struct mscclppConnInfo recv[MSCCLPP_MAX_CONNS];
};

// struct alignas(16) mscclppDevChannel {
//   struct mscclppDevChannelPeer *peers;
//   struct mscclppRing ring;
//   struct mscclppTree tree;
//   struct mscclppTree collnetChain;
//   struct mscclppDirect collnetDirect;
//   uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
// };

// struct mscclppDevComm {
//   int rank;
//   int nRanks;
//   int buffSizes[MSCCLPP_NUM_PROTOCOLS];

//   // Operation list for aggregation
//   int workFifoDepth;
//   struct mscclppWork* workFifoHeap; // may be cudaHost or GDR memory

//   // Flag to ask MSCCLPP kernels to abort
//   volatile uint32_t* abortFlag;

//   // Channels, device side
//   struct mscclppDevChannel* channels/*[MAXCHANNELS]*/;
// };

// struct alignas(16) mscclppDevCommAndChannels {
//   struct mscclppDevComm comm;
//   struct mscclppDevChannel channels[MAXCHANNELS];
// };

#endif
