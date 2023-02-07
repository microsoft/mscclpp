/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_PROXY_H_
#define MSCCLPP_PROXY_H_

// #include "devcomm.h"
// #include "info.h"
#include "socket.h"
// #include <pthread.h>
// #include "shm.h"

// enum mscclppProxyOpState { mscclppProxyOpNone, mscclppProxyOpReady, mscclppProxyOpProgress };

// struct mscclppProxyArgs;
// typedef mscclppResult_t (*proxyProgressFunc_t)(struct mscclppComm*, struct mscclppProxyArgs*);

// #define MSCCLPP_PROXY_MAX_SUBS MAXCHANNELS
// static_assert(MSCCLPP_MAX_WORK_ELEMENTS <= MAXCHANNELS, "Not enough sub space for max work elements");

// struct mscclppProxyOp {
//   struct mscclppProxyConnection* connection;
//   int channelId;
//   int nsteps;
//   ssize_t nbytes;
//   int root;
//   int next;

//   uint64_t opCount;
//   int sliceSteps;
//   int chunkSteps;
//   int chunkSize;
//   uint8_t /*mscclppDataType_t*/ dtype;
//   uint8_t /*mscclppDevRedOp_t*/ redOp;
//   uint8_t /*mscclppPattern_t*/ pattern;
//   uint8_t protocol;

//   union {
//     uint64_t unused;
//     // For use by enqueue.cc
//     struct mscclppProxyOp *enqNext;
//   };
// };
// static_assert(sizeof(struct mscclppProxyOp) == 64, "Keep ProxyOp aligned with cache lines for effective prefetch");

// struct mscclppProxySubArgs {
//   struct mscclppProxyConnection* connection;
//   int channelId;
//   int nsteps;
//   ssize_t nbytes;
//   int peer;

//   int groupSize; // Number of consecutive sub operations sharing the same recvComm
//   uint64_t base;
//   uint64_t posted;
//   uint64_t received;
//   uint64_t flushed;
//   uint64_t transmitted;
//   uint64_t done;
//   uint64_t end;
//   void* requests[MSCCLPP_STEPS];
//   void* profilingEvents[MSCCLPP_STEPS];
// };

// struct mscclppProxyArgs {
//   struct mscclppProxySubArgs subs[MSCCLPP_PROXY_MAX_SUBS];
//   proxyProgressFunc_t progress;
//   int nsubs;
//   int done;
//   uint64_t opCount;
//   int sliceSteps;
//   int chunkSteps;
//   int chunkSize;
//   uint8_t /*mscclppDataType_t*/ dtype;
//   uint8_t /*mscclppDevRedOp_t*/ redOp;
//   uint8_t /*mscclppPattern_t*/ pattern;
//   uint8_t protocol;
//   int state;
//   char* sharedBuff[MSCCLPP_STEPS];
//   int sharedSize[MSCCLPP_STEPS];

//   int idle;

//   // Element linking
//   struct mscclppProxyArgs* next;
//   struct mscclppProxyArgs* nextPeer;
//   struct mscclppProxyArgs** proxyAppendPtr;
// };
// #define MSCCLPP_MAX_NETDEVS 128

// // ProxyOps are used to communicate between main thread and service thread
// // Make sure we have enough to store two full rounds of operations on all channels.
// // Otherwise we'd be unable to post half of them to free new elements.
// #define MAX_OPS_PER_PEER (2*MAXCHANNELS*MSCCLPP_MAX_WORK_ELEMENTS_P2P)
#define MSCCLPP_MAX_LOCAL_RANKS 64
// struct mscclppProxyOpsPool {
//   struct mscclppProxyOp ops[MAX_OPS_PER_PEER*MSCCLPP_MAX_LOCAL_RANKS];
//   volatile int nextOps;
//   volatile int nextOpsEnd;
//   volatile int freeOps[MSCCLPP_MAX_LOCAL_RANKS];
//   pthread_mutex_t mutex;
//   pthread_cond_t cond;
// };

// struct mscclppProxyOps {
//   mscclppProxyOpsPool* pool;
//   mscclppShmHandle_t handle;
//   int count;
//   int freeOp;
//   int nextOps;
//   int nextOpsEnd;
// };

// struct mscclppProxySharedP2p {
//   int refcount;
//   int size;
//   char* cudaBuff;
//   char* hostBuff;
//   cudaIpcMemHandle_t ipc;
//   struct mscclppProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
// };

// struct mscclppProxySharedCollNet {
//   int size;
//   char* cudaBuff;
//   char* hostBuff;
//   struct mscclppProxyArgs* proxyAppend[2*MSCCLPP_MAX_NETDEVS];
//   void* resources;
// };

// struct mscclppProxyPeer {
//   struct mscclppProxySharedP2p send;
//   struct mscclppProxySharedP2p recv;
// };

// struct mscclppSharedNetComms {
//   void* sendComm[MAXCHANNELS];
//   void* recvComm[MAXCHANNELS];
//   int sendRefCount[MAXCHANNELS];
//   int recvRefCount[MAXCHANNELS];
// };

// struct mscclppProxyPool;
// struct mscclppProxyProgressState {
//   // Used by main threads to send work to progress thread
//   struct mscclppProxyOpsPool* opsPool;
//   mscclppShmHandle_t handle;
//   char opsPoolShmSuffix[6];

//   pthread_t thread;
//   bool stop;
//   struct mscclppProxyPeer** localPeers;
//   struct mscclppSharedNetComms* netComms[MSCCLPP_MAX_NETDEVS];
//   struct mscclppProxySharedCollNet collNet;
//   struct mscclppProxyArgs* active;
//   struct mscclppProxyArgs* pool;
//   struct mscclppProxyPool* pools;
//   int nextOps;
// };

struct mscclppProxyState {
  // Service thread
  pthread_t thread;
  struct mscclppSocket* listenSock;
  int stop;
//   CUcontext cudaCtx;

  // Used by main thread
  union mscclppSocketAddress* peerAddresses;
  struct mscclppSocket* peerSocks;
//   struct mscclppProxyOps* proxyOps;
//   void** sharedDevMems;

  // Progress thread
//   struct mscclppProxyProgressState progressState;
};

// enum proxyConnectState {
//   connUninitialized     = 0,
//   connInitialized       = 1,
//   connSharedInitialized = 2,
//   connSetupDone         = 3,
//   connConnected         = 4,
//   numConnStates         = 5
// };

// struct mscclppProxyConnection {
//   int send, transport, shared;
//   int localRank;
//   struct mscclppSocket* sock;
//   struct mscclppTransportComm* tcomm;
//   struct mscclppProxyArgs *proxyAppend;
//   struct mscclppProxyArgs **proxyAppendPtr;
//   void* transportResources;
//   proxyConnectState state;
// };

// typedef mscclppResult_t (*threadFunc_t)(struct mscclppProxyArgs*);

// enum proxyMode {
//   proxyRing = 0,
//   proxyFrom = 1,
//   proxyTo = 2
// };

// mscclppResult_t mscclppProxySaveOp(struct mscclppComm* comm, struct mscclppProxyOp* proxyOp, bool *justInquire);
// mscclppResult_t mscclppProxyComputeP2p(struct mscclppInfo* info, struct mscclppProxyOp* proxyOp);
// mscclppResult_t mscclppProxyStart(struct mscclppComm* comm);
mscclppResult_t mscclppProxyInit(struct mscclppComm* comm, struct mscclppSocket* sock, union mscclppSocketAddress* peerAddresses);
// mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
// mscclppResult_t mscclppProxyConnect(struct mscclppComm* comm, int transport, int send, int rank, struct mscclppProxyConnector* proxyConn);
// enum mscclppProxyMsgType {
//   mscclppProxyMsgInit = 1,
//   mscclppProxyMsgSharedInit = 2,
//   mscclppProxyMsgSetup = 3,
//   mscclppProxyMsgConnect = 4,
//   mscclppProxyMsgStart = 5,
//   mscclppProxyMsgClose = 6,
//   mscclppProxyMsgAbort = 7,
//   mscclppProxyMsgStop = 8
// };

// mscclppResult_t mscclppProxyCall(struct mscclppProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
// mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);
// mscclppResult_t mscclppProxyShmUnlink(struct mscclppComm* comm);
#endif
