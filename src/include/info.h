/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_INFO_H_
#define MSCCLPP_INFO_H_

#include "mscclpp.h"
#include "devcomm.h"
// #include "collectives.h"
#include "core.h"
#include "utils.h"
#include "strongstream.h"

// typedef enum : uint8_t {
//   mscclppPatternRing,
//   mscclppPatternRingTwice,
//   mscclppPatternPipelineFrom,
//   mscclppPatternPipelineTo,
//   mscclppPatternTreeUp,
//   mscclppPatternTreeDown,
//   mscclppPatternTreeUpDown,
//   mscclppPatternCollnetChain,
//   mscclppPatternCollnetDirect,
//   mscclppPatternSend,
//   mscclppPatternRecv
// } mscclppPattern_t;

// // Used to pass MSCCLPP call information between functions
// struct mscclppInfo {
//   mscclppFunc_t coll;
//   const char* opName;
//   // MSCCLPP Coll Args
//   const void* sendbuff;
//   void* recvbuff;
//   size_t count;
//   mscclppDataType_t datatype;
//   mscclppRedOp_t op;
//   int root; // peer for p2p operations
//   mscclppComm_t comm;
//   cudaStream_t stream;
//   // Algorithm details
//   int chunkSteps;
//   int sliceSteps;
//   // Computed later
//   mscclppDevRedOpFull opFull;
//   int algorithm;
//   int protocol;
//   mscclppPattern_t pattern;
//   int nChannels;
//   int nThreads;
//   size_t nBytes;
//   int nstepsPerLoop;
//   int nchunksPerLoop;
//   int chunkSize;
//   int channelId;
// };

// inline mscclppResult_t mscclppInfoSetDerived(struct mscclppInfo* info, int nRanks) {
//   info->nBytes = info->count * mscclppTypeSize(info->datatype);
//   if (info->coll == mscclppFuncAllGather || info->coll == mscclppFuncBroadcast) {
//     info->count = info->nBytes;
//     info->datatype = mscclppInt8;
//   }
//   if (info->coll == mscclppFuncAllGather || info->coll == mscclppFuncReduceScatter) info->nBytes *= nRanks; // count is per rank
//   return mscclppSuccess;
// }

// struct mscclppTaskColl {
//   struct mscclppTaskColl* next;
//   mscclppFunc_t func;
//   void const* sendbuff;
//   void* recvbuff;
//   size_t count;
//   int root;
//   mscclppDataType_t datatype;
//   mscclppDevRedOpFull op;
//   int chunkSteps, sliceSteps;
// };
// struct mscclppTaskP2p {
//   mscclppTaskP2p *next;
//   void *buff;
//   size_t bytes;
//   // Stateful chunk index. If a p2p gets "cut" over two plans this keeps track
//   // of where it left off.
//   int chunk;
// };

// struct mscclppCudaStreamList {
//   struct mscclppCudaStreamList *next;
//   cudaStream_t stream;
// };

// struct mscclppTasks {
//   struct Peer {
//     bool sendSeen, recvSeen;
//     struct mscclppIntruQueue<struct mscclppTaskP2p, &mscclppTaskP2p::next> sendQueue;
//     struct mscclppIntruQueue<struct mscclppTaskP2p, &mscclppTaskP2p::next> recvQueue;
//   };
//   struct mscclppIntruQueue<mscclppTaskColl, &mscclppTaskColl::next> collQueue;
//   size_t collBytesTotal;
//   struct Peer* peers/*[nRanks]*/;
//   int *p2pSendOrder/*[nRanks]*/, *p2pRecvOrder/*[nRanks]*/;
//   int nTasksColl, nTasksP2p;

//   // The list of user streams aggregated over all tasks present.
//   struct mscclppCudaStreamList* streams;
//   // The most recent user stream. Ignored if streams==nullptr
//   cudaStream_t streamRecent;
//   // The graph capturing all user streams or invalid if none. Thus we restrict the
//   // user that all streams must be captured in the same graph or not captured
//   // at all. Technically we could probably relax this, but that would mean
//   // collecting a different `mscclppTasks` per graph and one for non-graph.
//   struct mscclppCudaGraph capturingGraph;
// };

#endif
