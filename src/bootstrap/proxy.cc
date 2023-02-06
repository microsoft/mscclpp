/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
// #include "info.h"
// #include "collectives.h"
#include "socket.h"
// #include "shm.h"
// #include "profiler.h"
// #define ENABLE_TIMER 0
// #include "timer.h"

#include <sys/syscall.h>

// enum { proxyRecv=0, proxySend=1 };

// static bool NeedProxy(int type, int pattern, int root, struct mscclppRing* ring, int nranks) {
//   if (pattern == mscclppPatternRing || pattern == mscclppPatternRingTwice) return true;

//   /* In chains, one rank does not need a proxy. Let's figure out which one it is */
//   /* Which index in the reorganized rings should we compare root against */
//   const int myrank = 0, nextrank = 1, prevrank = nranks-1;
//   int index = pattern == mscclppPatternPipelineFrom ?
//       /*                            no recv /  no send    if root = */
//       /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
//       /* reduce */ (type == proxyRecv ? prevrank :   myrank );
//   int rank = ring->userRanks[index];
//   return (root != rank);
// }

// #define PROXYARGS_ALLOCATE_SIZE MSCCLPP_MAX_OPS
// struct mscclppProxyPool {
//   struct mscclppProxyPool *next;
//   struct mscclppProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
// };

// static mscclppResult_t allocateArgs(struct mscclppProxyProgressState* state, struct mscclppProxyArgs** argsptr) {
//   struct mscclppProxyArgs* elem;
//   if (state->pool == NULL) {
//     // Allocate a new pool of elements. Make sure we allocate the memory close
//     // to the network thread
//     struct mscclppProxyPool* newPool;
//     MSCCLPPCHECK(mscclppCalloc(&newPool, 1));

//     struct mscclppProxyArgs* newElems = newPool->elems;
//     // Chain newly allocated elements
//     for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
//       if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
//     }
//     // Add them all to the pool list
//     state->pool = newElems;
//     // Save the pool memory block for later resource release
//     newPool->next = state->pools;
//     state->pools = newPool;
//   }
//   elem = state->pool;
//   state->pool = state->pool->next;
//   elem->next = elem->nextPeer = NULL;
//   *argsptr = elem;
//   return mscclppSuccess;
// }

// //#define DEBUG_PROXY 1
// #ifdef DEBUG_PROXY
// #define DEBUG_PROXY_PRINT printf
// #else
// #define DEBUG_PROXY_PRINT(...)
// #endif

// #define OP_INDEX(op) ((op) ? (op)-state->pools->elems : -1)
// #define OP_SEEN 0x100000

// mscclppResult_t getOpIndex(struct mscclppProxyArgs* op, struct mscclppProxyProgressState* state, int* poolIndex, int* opIndex) {
//   struct mscclppProxyPool* pool = state->pools;
//   int p = 0;
//   while (pool) {
//     uint64_t o = op-pool->elems;
//     if (o < PROXYARGS_ALLOCATE_SIZE) {
//       *opIndex = o;
//       *poolIndex = p;
//       return mscclppSuccess;
//     }
//     pool = pool->next;
//     p++;
//   }
//   WARN("Could not find pool of op %p\n", op);
//   return mscclppInternalError;
// }

// mscclppResult_t printProxyOp(struct mscclppProxyArgs* op, int poolIndex, int opIndex) {
//   printf("[%d-%d|%ld| %s", poolIndex, opIndex, op->opCount, op->pattern == mscclppPatternSend ? "Send" : op->pattern == mscclppPatternRecv ? "Recv" : "Coll");
//   for (int s=0; s<op->nsubs; s++) {
//     struct mscclppProxySubArgs* sub = op->subs+s;
//     if (op->state == mscclppProxyOpProgress) {
//       char status = ' ';
//       if (op->pattern == mscclppPatternRecv) {
//         if (sub->posted < sub->nsteps && sub->posted < sub->done + MSCCLPP_STEPS) status = 'I'; // Init
//         else if (sub->received < sub->posted) status = 'R'; // Receiving
//         else if (sub->received < sub->transmitted) status = 'R'; // Receiving
//         else if (sub->transmitted < sub->received) status = 'F'; // Flushing
//         else if (sub->done < sub->transmitted) status = 'G'; // Waiting on GPU
//         else status = 'D'; // Done
//       } else if (op->pattern == mscclppPatternSend) {
//         if (sub->posted < sub->nsteps && sub->posted < sub->done + MSCCLPP_STEPS) status = 'I'; // Init
//         else if (sub->transmitted < sub->posted) status = 'G'; // Waiting on GPU
//         else if (sub->done < sub->transmitted) status = 'S'; // Sending
//         else status = 'D'; // Done
//       }
//       printf(" %d%c/%d", sub->peer, status, sub->channelId);
//     } else {
//       printf(" %d/%d", sub->peer, sub->channelId);
//     }
//   }
//   printf("]");
//   return mscclppSuccess;
// }
// mscclppResult_t dumpProxyState(struct mscclppProxyProgressState* state) {
//   struct mscclppProxyArgs* op = state->active;
//   int poolIndex, opIndex;
//   printf("ACTIVE OPS\n");
//   while (op) {
//     MSCCLPPCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
//     if (op->state & OP_SEEN) {
//       WARN("List loop at element %d-%d", poolIndex, opIndex);
//     }
//     MSCCLPPCHECK(printProxyOp(op, poolIndex, opIndex));
//     op->state |= OP_SEEN;
//     printf("\n");
//     struct mscclppProxyArgs* nextOp = op->nextPeer;
//     while (nextOp) {
//       MSCCLPPCHECK(getOpIndex(nextOp, state, &poolIndex, &opIndex));
//       if (nextOp->state & OP_SEEN) {
//         WARN("List loop at element %d-%d", poolIndex, opIndex);
//       }
//       printf("| `-> ");
//       MSCCLPPCHECK(printProxyOp(nextOp, poolIndex, opIndex));
//       nextOp->state |= OP_SEEN;
//       printf("\n");
//       if (nextOp->next) {
//         WARN("Inactive op has next set!\n");
//       }
//       nextOp = nextOp->nextPeer;
//     }
//     if (op->nextPeer == NULL) printf("|\n");
//     op = op->next;
//     printf("v\n");
//   }
//   printf("[X]\n");

// # if 0
//   printf("FREE OPS\n");
//   op = state->pool;
//   while (op) {
//     MSCCLPPCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
//     if (op->state & OP_SEEN) {
//       WARN("List loop at element %d-%d", poolIndex, opIndex);
//     }
//     MSCCLPPCHECK(printProxyOp(op, poolIndex, opIndex));
//     op->state |= OP_SEEN;
//     printf("->");
//     op = op->next;
//   }
//   printf("[X]\n");
// #else
//   op = state->pool;
//   while (op) {
//     MSCCLPPCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
//     if (op->state & OP_SEEN) {
//       WARN("List loop at element %d-%d", poolIndex, opIndex);
//     }
//     op->state |= OP_SEEN;
//     op = op->next;
//   }
// #endif

//   struct mscclppProxyPool* pool = state->pools;
//   poolIndex = 0;
//   while (pool) {
//     struct mscclppProxyArgs* elem = pool->elems;
//     for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++, elem++) {
//       if ((elem->state & OP_SEEN) == 0) {
//         printf("Elem %d-%d is not in any list:\n", poolIndex, e);
//         MSCCLPPCHECK(printProxyOp(elem, poolIndex, e));
//         printf("\n");
//       } else {
//         elem->state -= OP_SEEN;
//       }
//     }
//     pool = pool->next;
//     poolIndex++;
//   }
//   return mscclppSuccess;
// }

// static mscclppResult_t mscclppProxyOpToArgs(struct mscclppProxyOp* op, struct mscclppProxyArgs* args, int subIndex) {
//   struct mscclppProxySubArgs* sub = args->subs+subIndex;
//   if (subIndex >= MSCCLPP_PROXY_MAX_SUBS) {
//     WARN("Proxy append out of bounds");
//     return mscclppInternalError;
//   }

//   //memset(sub, 0, sizeof(struct mscclppProxySubArgs));
//   sub->connection = op->connection;
//   sub->channelId = op->channelId;
//   sub->nsteps = op->nsteps;
//   sub->nbytes = op->nbytes;
//   sub->peer = op->root;
//   args->nsubs = subIndex+1;
//   if (subIndex) {
//     if ((args->sliceSteps != op->sliceSteps) ||
//         (args->chunkSteps != op->chunkSteps) ||
//         (args->protocol != op->protocol) ||
//         (args->dtype != op->dtype) ||
//         (args->redOp != op->redOp)) {
//       WARN("Proxy append mismatch");
//       return mscclppInternalError;
//     }
//     if (args->state != mscclppProxyOpReady) {
//       WARN("Proxy append on running operation");
//       return mscclppInternalError;
//     }
//     return mscclppSuccess;
//   }
//   //memset(&args->progress, 0, sizeof(struct mscclppProxyArgs)-offsetof(struct mscclppProxyArgs, progress));
//   args->done = 0;
//   args->opCount = op->opCount;
//   args->sliceSteps = op->sliceSteps;
//   args->chunkSteps = op->chunkSteps;
//   args->chunkSize = op->chunkSize;
//   args->dtype = op->dtype;
//   args->redOp = op->redOp;
//   args->pattern = op->pattern;
//   args->protocol = op->protocol;
//   args->state = mscclppProxyOpReady;
//   args->progress = op->connection->tcomm->proxyProgress;
//   args->proxyAppendPtr = op->connection->proxyAppendPtr;
//   return mscclppSuccess;
// }

// static mscclppResult_t ProxyAppend(struct mscclppProxyProgressState* state, struct mscclppProxyOp* op) {
//   struct mscclppProxyConnection* connection = op->connection;
//   int shared = connection->shared;
//   struct mscclppProxyArgs* args = *connection->proxyAppendPtr;

//   if (args) {
//     if (shared && args->opCount == op->opCount) {
//       MSCCLPPCHECK(mscclppProxyOpToArgs(op, args, args->nsubs));
//       DEBUG_PROXY_PRINT("Insert (%d/%5ld/%5ld) as group with %5ld\n", shared, args->opCount, op->opCount, OP_INDEX(args));
//     } else {
//       struct mscclppProxyArgs* prevArgs = args;
//       MSCCLPPCHECK(allocateArgs(state, &args));
//       MSCCLPPCHECK(mscclppProxyOpToArgs(op, args, 0));
//       prevArgs->nextPeer = args;
//       DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, prevArgs->opCount, args->opCount, OP_INDEX(prevArgs));
//       *(args->proxyAppendPtr) = args;
//     }
//   } else {
//     // Nothing running for that peer. Add to the list
//     MSCCLPPCHECK(allocateArgs(state, &args));
//     MSCCLPPCHECK(mscclppProxyOpToArgs(op, args, 0));
//     if (state->active == NULL) {
//       // Create the list
//       DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
//       state->active = args;
//     } else {
//       // Append element at the end of the list
//       struct mscclppProxyArgs* last = state->active;
//       while (last->next) last = last->next;
//       last->next = args;
//       DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args), shared, args->opCount);
//     }
//     *(args->proxyAppendPtr) = args;
//   }
//   return mscclppSuccess;
// }

// mscclppResult_t mscclppProxyPost(struct mscclppProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
//   pthread_mutex_lock(&pool->mutex);
//   if (pool->nextOps == -1) {
//     pool->nextOps = nextOps;
//     pthread_cond_signal(&pool->cond);
//   } else {
//     pool->ops[pool->nextOpsEnd].next = nextOps;
//   }
//   pool->nextOpsEnd = nextOpsEnd;
//   pthread_mutex_unlock(&pool->mutex);
//   return mscclppSuccess;
// }

// mscclppResult_t mscclppLocalOpAppend(struct mscclppComm* comm, struct mscclppProxyConnector* proxyConn, struct mscclppProxyOp* proxyOp) {
//   struct mscclppProxyOps* proxyOps = proxyConn->comm->proxyState.proxyOps;
//   if (proxyOps == NULL) return mscclppInternalError;
//   proxyOps += proxyConn->localRank;
//   struct mscclppProxyOpsPool* pool = proxyOps->pool;

//   TIME_START(0);
//   int opIndex = proxyOps->freeOp;
//   struct mscclppProxyOp* op;
//   if (opIndex != -1) {
//     op = pool->ops+opIndex;
//     proxyOps->freeOp = op->next;
//   } else {
//     int freeOp;
//     while ((freeOp = pool->freeOps[comm->localRank]) == -1) sched_yield();
//     int freeOpNew;
//     while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+comm->localRank, freeOp, -1)) != freeOp) freeOp = freeOpNew;
//     opIndex = freeOp;
//     op = pool->ops+opIndex;
//     proxyOps->freeOp = op->next;
//   }
//   if (op->next != -1) __builtin_prefetch(pool->ops+op->next); // Prefetch next free op
//   memcpy(op, proxyOp, sizeof(struct mscclppProxyOp));
//   op->next = -1;
//   op->connection = proxyConn->connection;
//   if (proxyOps->nextOps == -1) {
//     proxyOps->nextOps = proxyOps->nextOpsEnd = opIndex;
//   } else {
//     pool->ops[proxyOps->nextOpsEnd].next = opIndex;
//     proxyOps->nextOpsEnd = opIndex;
//   }
//   if (++proxyOps->count == MAX_OPS_PER_PEER) {
//     // Post what we have so far to free some ops in the pool
//     // Do not post last operations as we could have more coming with the same opCount, and posting
//     // them in different batches would break proxyArgs aggregation with subs.
//     uint64_t lastOpCount = pool->ops[proxyOps->nextOpsEnd].opCount;
//     int lastOp = -1;
//     int toSend = 0;
//     int ops = 0;
//     for (int op= proxyOps->nextOps; op != proxyOps->nextOpsEnd; op=pool->ops[op].next) {
//       ops++;
//       if (pool->ops[op].opCount != lastOpCount) {
//         lastOp = op;
//         toSend = ops;
//       }
//     }
//     if (lastOp == -1) {
//       WARN("Unable to post incomplete proxy op chain %d..%d (opCount %ld)\n", proxyOps->nextOps, proxyOps->nextOpsEnd, lastOpCount);
//       return mscclppInternalError;
//     }
//     // Cut chain at lastOp
//     int nextOps = proxyOps->nextOps;
//     proxyOps->nextOps = pool->ops[lastOp].next;
//     pool->ops[lastOp].next = -1;
//     MSCCLPPCHECK(mscclppProxyPost(proxyOps->pool, nextOps, lastOp));
//     proxyOps->count -= toSend;
//   }
//   TIME_STOP(0);
//   return mscclppSuccess;
// }

// static mscclppResult_t SaveProxy(struct mscclppChannel* channel, int type, int peer, struct mscclppProxyOp* op, int connIndex, bool* justInquire) {
//   if (peer < 0) return mscclppSuccess;

//   struct mscclppChannelPeer* peerComm = channel->peers+peer;
//   struct mscclppConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
//   if (connector->transportComm == NULL) {
//     WARN("Rank %d has no transport for %s peer %d on channel %d/%d", connector->comm->rank,
//         type == proxyRecv ? "recv" : "send", peer, channel->id, connIndex);
//     return mscclppInternalError;
//   }
//   if (connector->transportComm->proxyProgress == NULL) return mscclppSuccess;

//   if (justInquire) *justInquire = true;
//   else {
//     MSCCLPPCHECK(mscclppLocalOpAppend(connector->comm, &connector->proxyConn, op));
//   }
//   return mscclppSuccess;
// }

// // justInquire != nullptr means don't actually do anything, just assertain need of
// // mscclppProxySaveOp for this op.
// mscclppResult_t mscclppProxySaveOp(struct mscclppComm* comm, struct mscclppProxyOp* op, bool* justInquire) {
//   struct mscclppChannel* channel = &comm->channels[op->channelId];
//   if (justInquire) *justInquire = false;
//   switch (op->pattern) {
//   case mscclppPatternRing:
//   case mscclppPatternRingTwice:
//   case mscclppPatternPipelineFrom:
//   case mscclppPatternPipelineTo: {
//       struct mscclppRing* ring = &channel->ring;
//       if (NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks)) {
//         MSCCLPPCHECK(SaveProxy(channel, proxyRecv, ring->prev, op, 0, justInquire));
//       }
//       if (NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks)) {
//         MSCCLPPCHECK(SaveProxy(channel, proxySend, ring->next, op, 0, justInquire));
//       }
//     } break;
//   case mscclppPatternTreeUp:
//   case mscclppPatternTreeDown:
//   case mscclppPatternTreeUpDown: {
//       if (op->pattern != mscclppPatternTreeDown) { // Tree up
//         struct mscclppTree* tree = &channel->tree;
//         for (int i=0; i<MSCCLPP_MAX_TREE_ARITY; i++) {
//           MSCCLPPCHECK(SaveProxy(channel, proxyRecv, tree->down[i], op, 0, justInquire));
//         }
//         MSCCLPPCHECK(SaveProxy(channel, proxySend, tree->up, op, 0, justInquire));
//       }
//       if (op->pattern != mscclppPatternTreeUp) { // Tree down
//         struct mscclppTree* tree = &channel->tree;
//         for (int i=0; i< MSCCLPP_MAX_TREE_ARITY; i++) {
//           MSCCLPPCHECK(SaveProxy(channel, proxySend, tree->down[i], op, 0, justInquire));
//         }
//         MSCCLPPCHECK(SaveProxy(channel, proxyRecv, tree->up, op, 0, justInquire));
//       }
//     } break;
//   case mscclppPatternCollnetChain: {
//       MSCCLPPCHECK(SaveProxy(channel, proxySend, channel->collnetChain.up, op, 1, justInquire));
//       MSCCLPPCHECK(SaveProxy(channel, proxyRecv, channel->collnetChain.up, op, 0, justInquire));
//     } break;
//   case mscclppPatternCollnetDirect: {
//       MSCCLPPCHECK(SaveProxy(channel, proxySend, channel->collnetDirect.out, op, 1, justInquire));
//       MSCCLPPCHECK(SaveProxy(channel, proxyRecv, channel->collnetDirect.out, op, 0, justInquire));
//     } break;
//   case mscclppPatternSend:
//   case mscclppPatternRecv: {
//       if (op->root == comm->rank) return mscclppSuccess;
//       MSCCLPPCHECK(SaveProxy(channel, op->pattern == mscclppPatternSend ? proxySend : proxyRecv, op->root, op, 1, justInquire));
//     } break;
//   }
//   return mscclppSuccess;
// }

// MSCCLPP_PARAM(ChunkSize, "CHUNK_SIZE", 0);

// mscclppResult_t mscclppProxyComputeP2p(struct mscclppInfo* info, struct mscclppProxyOp* op) {
//   memset(op, 0, sizeof(struct mscclppProxyOp));
//   int channelId = info->channelId;
//   struct mscclppChannel* channel = info->comm->channels+channelId;
//   op->channelId = channelId;
//   op->sliceSteps = 1;
//   op->chunkSteps = 1;
//   op->dtype = info->datatype;
//   op->protocol = info->protocol;

//   int stepSize = info->comm->buffSizes[op->protocol]/MSCCLPP_STEPS;

//   if (op->protocol == MSCCLPP_PROTO_SIMPLE) stepSize = info->comm->p2pChunkSize;
//   info->chunkSize = stepSize;
//   op->root = info->root;

//   struct mscclppChannelPeer* peer = channel->peers + op->root;
//   if (info->coll == mscclppFuncSend) {
//     op->pattern = mscclppPatternSend;
//     if (op->root != info->comm->rank && peer->send[1].transportComm == &netTransport.send) {
//       // Tune chunk size for the network
//       if (info->count < stepSize) info->chunkSize /= 4;
//       else if (info->count < 8*stepSize) info->chunkSize /= 2;
//     }
//   } else if (info->coll == mscclppFuncRecv) {
//     op->pattern = mscclppPatternRecv;
//     if (op->root != info->comm->rank && peer->recv[1].transportComm == &netTransport.recv) {
//       // Tune chunk size for the network
//       if (info->count < stepSize) info->chunkSize /= 4;
//       else if (info->count < 8*stepSize) info->chunkSize /= 2;
//     }
//   } else {
//     WARN("P2p operation is neither send or recv");
//     return mscclppInternalError;
//   }
//   if (mscclppParamChunkSize() != 0) {
//     info->chunkSize = mscclppParamChunkSize();
//   }
//   op->chunkSize = info->chunkSize;

//   // Compute nSteps for proxies
//   int chunkEffectiveSize = op->chunkSize;
//   if (op->protocol == MSCCLPP_PROTO_LL) {
//     chunkEffectiveSize /= 2;
//   }

//   op->nbytes = stepSize;
//   op->nsteps = DIVUP(info->count, chunkEffectiveSize);
//   if (op->nsteps == 0) op->nsteps = 1;

//   return mscclppSuccess;
// }

// static mscclppResult_t removeOp(struct mscclppProxyProgressState* state, struct mscclppProxyArgs** opPtr, struct mscclppProxyArgs** prevOpPtr) {
//   struct mscclppProxyArgs* freeOp = *opPtr;
//   struct mscclppProxyArgs* next = freeOp->next;
//   DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(next));
//   *opPtr = next;
//   if (freeOp->nextPeer) {
//     // replace op by nextPeer
//     struct mscclppProxyArgs* nextPeer = freeOp->nextPeer;
//     if (*prevOpPtr) {
//       (*prevOpPtr)->next = nextPeer;
//     } else {
//       state->active = nextPeer;
//     }
//     nextPeer->next = next;
//     *(prevOpPtr) = nextPeer;
//   } else {
//     *(freeOp->proxyAppendPtr) = NULL;
//     if (*prevOpPtr) {
//       (*prevOpPtr)->next = next;
//     } else {
//       state->active = next;
//     }
//   }
//   freeOp->next = state->pool;
//   state->pool = freeOp;
//   DEBUG_PROXY_PRINT("Removed %5ld (%5ld) : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
// #ifdef DEBUG_PROXY
//   MSCCLPPCHECK(dumpProxyState(state));
// #endif
//   return mscclppSuccess;
// }

// static mscclppResult_t progressOps(struct mscclppComm* comm, struct mscclppProxyProgressState* state, struct mscclppProxyArgs* opStart, int* idle) {
//   struct mscclppProxyArgs* prevOp = NULL;
//   struct mscclppProxyArgs* op = opStart;
//   while (op) {
//     if (op->state == mscclppProxyOpNone) return mscclppInternalError;
//     TIME_START(0); TIME_START(1);
//     MSCCLPPCHECK(op->progress(comm, op));
//     if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } else { TIME_CANCEL(1); TIME_STOP(0); }
//     *idle &= op->idle;
//     if (op->state == mscclppProxyOpNone) {
//       TIME_START(2);
//       MSCCLPPCHECK(removeOp(state, &op, &prevOp));
//       TIME_STOP(2);
//     } else {
//       prevOp = op;
//       op = op->next;
//     }
//   }
//   return mscclppSuccess;
// }

// MSCCLPP_PARAM(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);

// static mscclppResult_t mscclppProxyGetPostedOps(struct mscclppComm* comm, int* added) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   if (state->opsPool == NULL) return mscclppInternalError;
//   struct mscclppProxyOpsPool* pool = state->opsPool;

//   struct mscclppProxyArgs profArgs; // Only used for profiling purposes
//   if (state->nextOps != -1) goto process_nextops;

//   // If we have ops to progress, no need to block waiting for something to arrive or even wait for the lock
//   // to be available. Exit, continue progress, and come back later.
//   if (state->active != NULL && (pool->nextOps == -1 || pthread_mutex_trylock(&pool->mutex) != 0)) return mscclppSuccess;

//   if (state->active == NULL) {
//     pthread_mutex_lock(&pool->mutex);
//     while (pool->nextOps == -1 && !state->stop) {
//       struct mscclppProxyArgs profArgs; // Only used for profiling purposes
//       mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileSleep);
//       pthread_cond_wait(&pool->cond, &pool->mutex);
//       mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileWakeup);
//     }
//     if (state->stop) { // We might have been woken up to stop.
//       pthread_mutex_unlock(&pool->mutex);
//       return mscclppSuccess;
//     }
//   }

//   state->nextOps = pool->nextOps;
//   pool->nextOps = pool->nextOpsEnd = -1;
//   pthread_mutex_unlock(&pool->mutex);
//   if (state->nextOps == -1) return mscclppInternalError;

// process_nextops:
//   mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileAppend);
//   TIME_START(2);
//   int freeOp[MSCCLPP_MAX_LOCAL_RANKS];
//   int freeOpEnd[MSCCLPP_MAX_LOCAL_RANKS];
//   for (int i=0; i<comm->localRanks; i++) freeOp[i] = -1;

//   uint64_t lastOpCount = 0;
//   int lastPeer = -1;
//   int count = 0;
//   for (int opIndex = state->nextOps; opIndex != -1;) {
//     struct mscclppProxyOp* peerOp = pool->ops+opIndex;
//     int peer = opIndex / MAX_OPS_PER_PEER;
//     if ((lastOpCount && peerOp->opCount != lastOpCount) || ((lastPeer != -1) && peer != lastPeer)) count++;
//     if (count == mscclppParamProxyAppendBatchSize()+1) break;
//     lastOpCount = peerOp->opCount;
//     lastPeer = peer;
//     if (peerOp->connection == NULL) return mscclppInternalError;
//     if (peerOp->next != -1) __builtin_prefetch(pool->ops+peerOp->next);
//     MSCCLPPCHECK(ProxyAppend(state, peerOp));
//     (*added)++;
//     int lastOpIndex = opIndex;
//     opIndex = peerOp->next;
//     // Return op to peer pool
//     if (freeOp[peer] == -1) {
//       freeOpEnd[peer] = lastOpIndex;
//     } else {
//       peerOp->next = freeOp[peer];
//     }
//     freeOp[peer] = lastOpIndex;
//     state->nextOps = opIndex;
//   }

//   for (int i=0; i<comm->localRanks; i++) {
//     if (freeOp[i] == -1) continue;
//     int newFree = freeOp[i];
//     int oldFree = pool->freeOps[i];
//     pool->ops[freeOpEnd[i]].next = oldFree;
//     if (oldFree == -1) {
//       // Nothing for the main thread to consume, we can set it.
//       pool->freeOps[i] = newFree;
//     } else {
//       // The main thread may recycle free ops at any time, replace the freeOps value atomically and check it worked.
//       int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
//       if (swap != oldFree) {
//         if (swap != -1) return mscclppInternalError;
//         // Ops were recycled while we were trying to swap, just set the value directly now.
//         pool->ops[freeOpEnd[i]].next = -1;
//         pool->freeOps[i] = newFree;
//       }
//     }
//   }
//   profArgs.opCount = *added;
//   mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileAppendEnd);
//   TIME_STOP(2);
//   return mscclppSuccess;
// }

// #include <signal.h>
// static mscclppProxyProgressState* mscclppLastProxyState;
// void mscclppDumpProxyState(int signal) {
//   dumpProxyState(mscclppLastProxyState);
// }

// MSCCLPP_PARAM(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
// mscclppResult_t mscclppSetThreadContext(struct mscclppComm* comm) {
// #if CUDART_VERSION >= 11030
//   static int createThreadContext = -1;

//   if (createThreadContext == -1) {
//     createThreadContext = mscclppParamCreateThreadContext();
//     if (createThreadContext) {
//       if (CUPFN(cuCtxCreate) == nullptr || CUPFN(cuCtxDestroy) == nullptr || CUPFN(cuCtxSetCurrent) == nullptr) {
//         WARN("Unable to create thread context due to old driver, disabling.");
//         createThreadContext = 0;
//       }
//     }
//   }
//   if (createThreadContext) {
//     if (comm->proxyState.cudaCtx == NULL) {
//       if (CUPFN(cuCtxCreate(&comm->proxyState.cudaCtx,
//                                   CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST, comm->cudaDev)) != CUDA_SUCCESS) {
//         WARN("Failed to create CUDA context on device %d", comm->cudaDev);
//         createThreadContext = 0;
//         return mscclppSuccess;
//       }
//     } else {
//       if (CUPFN(cuCtxSetCurrent(comm->proxyState.cudaCtx)) != CUDA_SUCCESS) {
//         WARN("Failed to set CUDA context on device %d", comm->cudaDev);
//         return mscclppUnhandledCudaError;
//       }
//     }
//   }
// #endif
//   return mscclppSuccess;
// }

// // Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
// MSCCLPP_PARAM(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);

// void* mscclppProxyProgress(void *comm_) {
//   struct mscclppComm* comm = (struct mscclppComm*)comm_;
//   if (mscclppSetThreadContext(comm) != mscclppSuccess) {
//     WARN("[Proxy Progress] Failed to set CUDA context on device %d", comm->cudaDev);
//   } else if (cudaSetDevice(comm->cudaDev) != cudaSuccess) {
//     WARN("[Proxy Progress] Failed to set CUDA device %d", comm->cudaDev);
//   }
//   if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);

//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   state->nextOps = -1;
//   const int sig = mscclppParamProxyDumpSignal();
//   if (sig != -1) signal(sig, mscclppDumpProxyState);
//   mscclppLastProxyState = state;
//   char threadName[MSCCLPP_THREAD_NAMELEN];
//   snprintf(threadName, MSCCLPP_THREAD_NAMELEN, "MSCCLPP Progress%2d", comm->cudaDev);
//   nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

//   int lastIdle = 0;
//   struct mscclppProxyArgs profArgs; // Only used for profiling purposes
//   while ((state->stop == false || (state->stop == true && state->active)) && *comm->abortFlag == 0) {
//     int idle = 1;
//     mscclppResult_t ret = progressOps(comm, state, state->active, &idle);
//     if (ret != mscclppSuccess) {
//       (void) mscclppCommSetAsyncError(comm, ret);
//       INFO(MSCCLPP_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
//       return NULL;
//     }
//     if (lastIdle == 0 && idle == 1) mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileIdle);
//     if (lastIdle == 1 && idle == 0) mscclppProfilingRecord(&profArgs, 0, 0, mscclppProxyProfileActive);
//     int added = 0;
//     TIME_START(3);
//     if (state->stop == false)
//       ret = mscclppProxyGetPostedOps(comm, &added);
//     if (added) { TIME_STOP(3); } else { TIME_CANCEL(3); }
//     if (ret != mscclppSuccess) {
//       (void) mscclppCommSetAsyncError(comm, ret);
//       INFO(MSCCLPP_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
//     }
//     if (added == 0) {
//       sched_yield(); // No request progressed. Let others run.
//     }
//     lastIdle = idle;
//   }
//   return NULL;
// }

// mscclppResult_t mscclppProxyStart(struct mscclppComm* comm) {
//   struct mscclppProxyOps* proxyOps = comm->proxyState.proxyOps;
//   if (proxyOps == NULL) return mscclppSuccess;
//   TIME_START(1);
//   for (int r=0; r<comm->localRanks; r++) {
//     struct mscclppProxyOps* ops = proxyOps+r;
//     if (ops->pool == NULL || ops->nextOps == -1) continue;
//     MSCCLPPCHECK(mscclppProxyPost(ops->pool, ops->nextOps, ops->nextOpsEnd));
//     ops->nextOps = ops->nextOpsEnd = -1;
//     ops->count = 0;
//   }
//   comm->opCount++;
//   TIME_STOP(1);
//   return mscclppSuccess;
// }

// mscclppResult_t mscclppProxyProgressCreate(struct mscclppComm* comm) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   if (!state->thread) {
//     pthread_create(&state->thread, NULL, mscclppProxyProgress, comm);
//     mscclppSetThreadName(state->thread, "MSCCLPP Progress%2d", comm->cudaDev);
//   }
//   return mscclppSuccess;
// }

// mscclppResult_t mscclppProxyProgressDestroy(struct mscclppComm* comm) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;

//   // Request the proxy to stop and then wake it
//   if (state->opsPool) {
//     pthread_mutex_lock(&state->opsPool->mutex);
//     state->stop = true;
//     pthread_cond_signal(&state->opsPool->cond);
//     pthread_mutex_unlock(&state->opsPool->mutex);
//     pthread_join(state->thread, NULL);
//   }

//   // Free off any memory allocated for the proxy arg pools
//   while (state->pools != NULL) {
//     struct mscclppProxyPool *next = state->pools->next;
//     free(state->pools);
//     state->pools = next;
//   }

//   mscclppProfilingDump();
//   TIME_PRINT("Proxy");
//   return mscclppSuccess;
// }

// struct mscclppProxyAsyncOp {
//   int type;
//   struct mscclppProxyConnection* connection;
//   int reqSize, respSize;
//   char *reqBuff, *respBuff;
// };

// struct mscclppProxyLocalPeer {
//   struct mscclppSocket sock;
//   int localRank;
//   struct mscclppProxyAsyncOp asyncOps;
// };

// #define MSCCLPP_PROXY_CONN_POOL_SIZE_POW2 7
// #define MSCCLPP_PROXY_CONN_POOL_SIZE (1<<(MSCCLPP_PROXY_CONN_POOL_SIZE_POW2))
// #define MSCCLPP_PROXY_CONN_POOL_MASK ((MSCCLPP_PROXY_CONN_POOL_SIZE)-1)
// struct mscclppProxyConnectionPool {
//   struct mscclppProxyConnection** pools;
//   int banks;
//   int offset;
//   struct mscclppProxyAsyncOp* ops;
// };

// static mscclppResult_t mscclppProxyNewConnection(struct mscclppProxyConnectionPool* pool, int* id) {
//   if (pool->offset == MSCCLPP_PROXY_CONN_POOL_SIZE) {
//     MSCCLPPCHECK(mscclppRealloc(&pool->pools, pool->banks, pool->banks+1));
//     MSCCLPPCHECK(mscclppCalloc(pool->pools+pool->banks, MSCCLPP_PROXY_CONN_POOL_SIZE));
//     pool->banks++;
//     pool->offset = 0;
//   }
//   *id = ((pool->banks-1) << MSCCLPP_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
//   pool->offset++;
//   return mscclppSuccess;
// }

// static mscclppResult_t mscclppProxyGetConnection(struct mscclppProxyConnectionPool* pool, int id, struct mscclppProxyConnection** conn) {
//   int bank = id>>MSCCLPP_PROXY_CONN_POOL_SIZE_POW2;
//   int offset = id&MSCCLPP_PROXY_CONN_POOL_MASK;
//   if ((pool->pools == NULL) || (bank > pool->banks) || (pool->pools[bank] == NULL)) return mscclppInternalError;
//   *conn = pool->pools[bank]+offset;
//   return mscclppSuccess;
// }

// static mscclppResult_t proxyFree(struct mscclppProxyConnection* connection, struct mscclppComm* comm) {
//   if (connection->send) {
//     if (mscclppTransports[connection->transport]->send.proxyFree) {
//       MSCCLPPCHECK(mscclppTransports[connection->transport]->send.proxyFree(connection, comm));
//     }
//   } else {
//     if (mscclppTransports[connection->transport]->recv.proxyFree) {
//       MSCCLPPCHECK(mscclppTransports[connection->transport]->recv.proxyFree(connection, comm));
//     }
//   }
//   return mscclppSuccess;
// }

// static mscclppResult_t mscclppProxyFreeConnections(struct mscclppProxyConnectionPool* pool, struct mscclppComm* comm) {
//   for (int b=0; b<pool->banks; b++) {
//     int max = b == pool->banks-1 ? pool->offset : MSCCLPP_PROXY_CONN_POOL_SIZE;
//     for (int i=0; i<max; i++) {
//       mscclppProxyConnection *connection = pool->pools[b]+i;
//       if (connection->state != connUninitialized) {
//         MSCCLPPCHECK(proxyFree(connection, comm));
//       }
//     }
//     free(pool->pools[b]);
//   }
//   free(pool->pools);
//   return mscclppSuccess;
// }

// #include "transport.h"

// mscclppResult_t mscclppProxyConnect(struct mscclppComm* comm, int transport, int send, int rank, struct mscclppProxyConnector* proxyConn) {
//   struct mscclppSocket* sock;
//   int ready;
//   int type = mscclppProxyMsgInit;

//   // Keep one connection per mlocal rank
//   proxyConn->connection = NULL;
//   proxyConn->rank = rank;
//   if (comm->proxyState.peerSocks == NULL) {
//     MSCCLPPCHECK(mscclppCalloc(&comm->proxyState.peerSocks, comm->localRanks));
//     MSCCLPPCHECK(mscclppCalloc(&comm->proxyState.proxyOps, comm->localRanks));
//     MSCCLPPCHECK(mscclppCalloc(&comm->proxyState.sharedDevMems, comm->localRanks));
//     for (int i = 0; i < comm->localRanks; ++i) {
//       MSCCLPPCHECK(mscclppSocketSetFd(-1, &comm->proxyState.peerSocks[i]));
//     }
//   }

//   MSCCLPPCHECK(mscclppTopoGetLocalRank(comm->topo, rank, &proxyConn->localRank));
//   sock = comm->proxyState.peerSocks + proxyConn->localRank;
//   MSCCLPPCHECK(mscclppSocketReady(sock, &ready));
//   if (!ready) {
//     MSCCLPPCHECK(mscclppSocketInit(sock, comm->proxyState.peerAddresses+rank, comm->magic, mscclppSocketTypeProxy, comm->abortFlag));
//     MSCCLPPCHECK(mscclppSocketConnect(sock));
//   }
//   MSCCLPPCHECK(mscclppSocketSend(sock, &type, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketSend(sock, &transport, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketSend(sock, &send, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketSend(sock, &comm->localRank, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &proxyConn->connection, sizeof(void*)));
//   struct mscclppTransportComm* tcomm = send ? &mscclppTransports[transport]->send : &mscclppTransports[transport]->recv;
//   // If we need proxy progress, map progress ops
//   if (tcomm->proxyProgress) {
//     char poolPath[] = "/dev/shm/mscclpp-XXXXXX";
//     MSCCLPPCHECK(mscclppSocketRecv(sock, poolPath+sizeof("/dev/shm/mscclpp-")-1, sizeof("XXXXXX")-1));
//     struct mscclppProxyOps* proxyOps = comm->proxyState.proxyOps+proxyConn->localRank;
//     if (proxyOps->pool == NULL) {
//       MSCCLPPCHECK(mscclppShmOpen(poolPath, sizeof(struct mscclppProxyOpsPool), (void**)(&proxyOps->pool), NULL, -1, &proxyOps->handle));
//       proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
//     }
//   }
//   INFO(MSCCLPP_NET, "Connection to proxy localRank %d -> connection %p", proxyConn->localRank, proxyConn->connection);
//   proxyConn->comm = comm;
//   return mscclppSuccess;
// }

// const char* mscclppProxyMsgTypeStr[] = { "Unknown", "Init", "SharedInit", "Setup", "Connect", "Start", "Close", "Abort", "Stop" };
// mscclppResult_t mscclppProxyCall(struct mscclppProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
//   struct mscclppSocket* sock;
//   mscclppResult_t ret = mscclppSuccess;

//   if (proxyConn->comm->proxyState.peerSocks == NULL) return mscclppInternalError;
//   sock = proxyConn->comm->proxyState.peerSocks + proxyConn->localRank;
//   if (sock == NULL) return mscclppInternalError;
//   MSCCLPPCHECKGOTO(mscclppSocketSend(sock, &type, sizeof(int)), ret, error);
//   MSCCLPPCHECKGOTO(mscclppSocketSend(sock, &proxyConn->connection, sizeof(void*)), ret, error);
//   MSCCLPPCHECKGOTO(mscclppSocketSend(sock, &reqSize, sizeof(int)), ret, error);
//   MSCCLPPCHECKGOTO(mscclppSocketSend(sock, &respSize, sizeof(int)), ret, error);
//   if (reqSize) MSCCLPPCHECKGOTO(mscclppSocketSend(sock, reqBuff, reqSize), ret, error);
//   if (respSize) MSCCLPPCHECKGOTO(mscclppSocketRecv(sock, respBuff, respSize), ret, error);
//   return mscclppSuccess;
// error:
//   WARN("Proxy Call to rank %d failed (%s)", proxyConn->comm->localRankToRank[proxyConn->localRank], mscclppProxyMsgTypeStr[type]);
//   return ret;
// }

// static mscclppResult_t proxyProgressInit(struct mscclppComm* comm) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   if (state->opsPool == NULL) {
//     int size = sizeof(struct mscclppProxyOpsPool);
//     struct mscclppProxyOpsPool* pool = NULL;

//     // The service thread may be launched already but localRanks may not be set yet.
//     while (comm->localRanks == 0) sched_yield();

//     char shmPath[sizeof("/dev/shm/mscclpp-XXXXXX")];
//     shmPath[0] = '\0';
//     MSCCLPPCHECK(mscclppShmOpen(shmPath, size, (void**)&pool, NULL, comm->localRanks + 1, &state->handle));
//     // Init pool
//     pool->nextOps = -1;

//     for (int r=0; r<comm->localRanks; r++) {
//       pool->freeOps[r] = r*MAX_OPS_PER_PEER;
//       for (int i=0; i<MAX_OPS_PER_PEER-1; i++) pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
//       pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
//     }

//     // Setup mutex/cond to work inter-process
//     pthread_mutexattr_t mutexAttr;
//     pthread_mutexattr_init(&mutexAttr);
//     pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
//     pthread_mutex_init(&pool->mutex, &mutexAttr);
//     pthread_condattr_t condAttr;
//     pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
//     pthread_cond_init(&pool->cond, &condAttr);
//     state->opsPool = pool;

//     memcpy(state->opsPoolShmSuffix, shmPath+sizeof("/dev/shm/mscclpp-")-1, sizeof("XXXXXX")-1);

//     // All ops structures are created, we can start the progress thread
//     MSCCLPPCHECK(mscclppProxyProgressCreate(comm));
//   }
//   return mscclppSuccess;
// }

// static void proxyOpsFree(struct mscclppComm* comm) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   if (mscclppShmClose(state->handle) != mscclppSuccess) {
//     WARN("[Service thread] shm close failed");
//   }
// }

// mscclppResult_t mscclppProxyShmUnlink(struct mscclppComm* comm) {
//   struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//   if (state->opsPool == NULL) return mscclppSuccess;

//   if (mscclppShmUnlink(state->handle) != mscclppSuccess) {
//     WARN("[Service thread] proxy ops shm unlink failed");
//   }
//   return mscclppSuccess;
// }

// static mscclppResult_t proxyConnInit(struct mscclppProxyLocalPeer* peer, struct mscclppProxyConnectionPool* connectionPool, struct mscclppComm* comm) {
//   struct mscclppSocket* sock = &peer->sock;
//   int id;
//   struct mscclppProxyConnection* connection;
//   MSCCLPPCHECK(mscclppProxyNewConnection(connectionPool, &id));
//   MSCCLPPCHECK(mscclppProxyGetConnection(connectionPool, id, &connection));
//   connection->sock = sock;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &connection->transport, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &connection->send, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &peer->localRank, sizeof(int)));
//   connection->localRank = peer->localRank;
//   MSCCLPPCHECK(mscclppSocketSend(sock, &connection, sizeof(void*)));
//   connection->tcomm = connection->send ? &mscclppTransports[connection->transport]->send : &mscclppTransports[connection->transport]->recv;
//   // If we need proxy progress, let's allocate ops and start the thread
//   if (connection->tcomm->proxyProgress) {
//     MSCCLPPCHECK(proxyProgressInit(comm));
//     struct mscclppProxyProgressState* state = &comm->proxyState.progressState;
//     MSCCLPPCHECK(mscclppSocketSend(sock, state->opsPoolShmSuffix, sizeof("XXXXXX")-1));
//   }
//   INFO(MSCCLPP_NET, "New proxy %s connection %d from local rank %d, transport %d", connection->send ? "send":"recv", id, connection->localRank, connection->transport);
//   __atomic_store_n(&connection->state, connInitialized, __ATOMIC_RELEASE);
//   return mscclppSuccess;
// }

// static mscclppResult_t proxyConnSharedInit(struct mscclppProxyLocalPeer* peer, struct mscclppProxyConnectionPool* connectionPool, struct mscclppComm* comm) {
//   struct mscclppSocket* sock = &peer->sock;
//   struct mscclppProxyConnection* connection;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &connection, sizeof(void*)));
//   int reqSize, respSize;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &reqSize, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &respSize, sizeof(int)));
//   if (reqSize != sizeof(int) || respSize != 0) return mscclppInternalError;
//   int nChannels;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &nChannels, sizeof(int)));
//   if (connection->tcomm->proxySharedInit) MSCCLPPCHECK(connection->tcomm->proxySharedInit(connection, comm, nChannels));
//   __atomic_store_n(&connection->state, connSharedInitialized, __ATOMIC_RELEASE);
//   return mscclppSuccess;
// }

// static mscclppResult_t proxyProgressAsync(struct mscclppProxyAsyncOp* op, struct mscclppComm* comm, int* asyncOpCount) {
//   int done = 1;
//   if (op->type == mscclppProxyMsgSetup) {
//     MSCCLPPCHECK(op->connection->tcomm->proxySetup(op->connection, comm, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done));
//   } else if (op->type == mscclppProxyMsgConnect) {
//     MSCCLPPCHECK(op->connection->tcomm->proxyConnect(op->connection, comm, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done));
//   } else return mscclppInternalError;
//   if (done) {
//     if (op->type == mscclppProxyMsgSetup)
//       __atomic_store_n(&op->connection->state, connSetupDone, __ATOMIC_RELEASE);
//     else if (op->type == mscclppProxyMsgConnect)
//       __atomic_store_n(&op->connection->state, connConnected, __ATOMIC_RELEASE);
//     /* if setup or connect is done, we should not return any error at this point since
//      * mscclppSocketSend might already send the respBuff to the requester. If we still choose
//      * to abort and close the connection, it can cause segfault if the requester is using
//      * the respBuff. */
//     if (op->respSize) mscclppSocketSend(op->connection->sock, op->respBuff, op->respSize);
//     if (op->reqBuff) {
//       free(op->reqBuff);
//       op->reqBuff = NULL;
//     }
//     if (op->respBuff) {
//       free(op->respBuff);
//       op->respBuff = NULL;
//     }
//     op->type = 0;
//     (*asyncOpCount)--;
//   } else if (*comm->abortFlag != 0) {
//     return mscclppInternalError;
//   }

//   return mscclppSuccess;
// }

// static mscclppResult_t proxyConnSetupConnect(int type, struct mscclppProxyLocalPeer* peer, struct mscclppProxyConnectionPool* connectionPool, struct mscclppComm* comm, int* asyncOpCount) {
//   struct mscclppSocket* sock = &peer->sock;
//   struct mscclppProxyAsyncOp* asyncOp = &peer->asyncOps;
//   asyncOp->type = type;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &asyncOp->connection, sizeof(void*)));

//   MSCCLPPCHECK(mscclppSocketRecv(sock, &asyncOp->reqSize, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &asyncOp->respSize, sizeof(int)));
//   if (asyncOp->reqSize) {
//     MSCCLPPCHECK(mscclppCalloc(&asyncOp->reqBuff, asyncOp->reqSize));
//     MSCCLPPCHECK(mscclppSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize));
//   }
//   if (asyncOp->respSize) MSCCLPPCHECK(mscclppCalloc(&asyncOp->respBuff, asyncOp->respSize));
//   (*asyncOpCount)++;
//   MSCCLPPCHECK(proxyProgressAsync(asyncOp, comm, asyncOpCount));
//   return mscclppSuccess;
// }

// #include <poll.h>

// void* mscclppProxyService(void* _args) {
//   struct mscclppComm* comm =  (struct mscclppComm *) _args;
//   if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
//   if (mscclppSetThreadContext(comm) != mscclppSuccess) {
//     WARN("[Proxy Service] Failed to set CUDA context on device %d", comm->cudaDev);
//   } else if (cudaSetDevice(comm->cudaDev) != cudaSuccess) {
//     WARN("[Proxy Service] Failed to set CUDA device %d", comm->cudaDev);
//   }
//   if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);

//   // Prepare poll descriptor
//   struct mscclppProxyConnectionPool connectionPool;
//   connectionPool.pools = NULL;
//   connectionPool.banks = 0;
//   connectionPool.offset = MSCCLPP_PROXY_CONN_POOL_SIZE;

//   struct pollfd pollfds[MSCCLPP_MAX_LOCAL_RANKS+1];
//   struct mscclppProxyLocalPeer peers[MSCCLPP_MAX_LOCAL_RANKS];
//   memset(&peers, 0, sizeof(struct mscclppProxyLocalPeer)*MSCCLPP_MAX_LOCAL_RANKS);
//   for (int s=0; s<MSCCLPP_MAX_LOCAL_RANKS; s++) {
//     pollfds[s].fd = -1;
//     pollfds[s].events = POLLHUP|POLLIN;
//   }
//   if (mscclppSocketGetFd(comm->proxyState.listenSock, &pollfds[MSCCLPP_MAX_LOCAL_RANKS].fd) != mscclppSuccess) {
//     WARN("[Proxy Service] Get listenSock fd fails\n");
//     return NULL;
//   };
//   pollfds[MSCCLPP_MAX_LOCAL_RANKS].events = POLLIN;

//   int maxnpeers = 0;
//   int npeers = 0;
//   int stop = 0;
//   int asyncOpCount = 0;
//   while (stop == 0 || (stop == 1 && npeers > 0)) {
//     /* Even if local comm aborts, we cannot let proxy thread exit if we still have peer
//      * connections. Need to wait until all other related comms call abort and safely exit
//      * together, or we could face segmentation fault. */
//     if (*comm->abortFlag != 0) stop = 1;
//     /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
//     int ret;
//     do {
//       ret = poll(pollfds, MSCCLPP_MAX_LOCAL_RANKS+1, asyncOpCount ? 0 : 500);
//     } while (ret < 0 && errno == EINTR);
//     if (ret < 0) {
//       WARN("[Proxy Service] Poll failed: %s", strerror(errno));
//       return NULL;
//     }
//     if (pollfds[MSCCLPP_MAX_LOCAL_RANKS].revents) {
//       int s = 0;
//       while (s < MSCCLPP_MAX_LOCAL_RANKS && pollfds[s].fd >= 0) s++;
//       if (s == MSCCLPP_MAX_LOCAL_RANKS) {
//         WARN("[Proxy service] Too many connections (%d max)", MSCCLPP_MAX_LOCAL_RANKS);
//         return NULL;
//       }
//       if (maxnpeers < s+1) maxnpeers = s+1;
//       if (mscclppSocketInit(&peers[s].sock) != mscclppSuccess) {
//         WARN("[Service thread] Initialize peers[%d].sock fails\n", s);
//         return NULL;
//       }
//       if (mscclppSocketAccept(&peers[s].sock, comm->proxyState.listenSock) != mscclppSuccess) {
//         WARN("[Service thread] Accept failed %s", strerror(errno));
//       } else {
//         if (mscclppSocketGetFd(&peers[s].sock, &pollfds[s].fd) != mscclppSuccess) {
//           WARN("[Service thread] Get peers[%d].sock fd fails\n", s);
//           return NULL;
//         }
//         npeers++;
//         peers[s].localRank = -1;
//       }
//     }
//     for (int s=0; s<maxnpeers; s++) {
//       struct mscclppProxyLocalPeer* peer = peers+s;
//       struct mscclppSocket* sock = &peer->sock;
//       struct mscclppProxyAsyncOp* op = &peer->asyncOps;
//       int closeConn = 0;
//       int type = 0;
//       mscclppResult_t res = mscclppSuccess;

//       if (pollfds[s].fd == -1) continue;
//       if (op->type != 0) {
//         res = proxyProgressAsync(op, comm, &asyncOpCount);
//         type = op->type;
//         if (res != mscclppSuccess) closeConn = 1;
//       } else if (pollfds[s].revents & POLLIN) {
//         int closed;
//         if (mscclppSocketTryRecv(sock, &type, sizeof(int), &closed) != mscclppSuccess) {
//           WARN("[Service thread] Could not receive type from localRank %d", peer->localRank);
//           closeConn = 1;
//         } else if (closed) {
//           INFO(MSCCLPP_INIT|MSCCLPP_NET, "[Service thread] Connection closed by localRank %d", peer->localRank);
//           closeConn = 1;
//         } else {
//           if (type == mscclppProxyMsgStop) {
//             stop = 1;
//             closeConn = 1;
//           } else if (type == mscclppProxyMsgClose) {
//             closeConn = 1;
//           } else if (type == mscclppProxyMsgInit) {
//             res = proxyConnInit(peers+s, &connectionPool, comm);
//           } else if (type == mscclppProxyMsgSharedInit) {
//             res = proxyConnSharedInit(peers+s, &connectionPool, comm);
//           } else if (type == mscclppProxyMsgSetup || type == mscclppProxyMsgConnect) {
//             res = proxyConnSetupConnect(type, peers+s, &connectionPool, comm, &asyncOpCount);
//           } else {
//             WARN("[Service thread] Unknown command %d from localRank %d\n", type, peer->localRank);
//             closeConn = 1;
//           }
//         }
//       } else if (pollfds[s].revents & POLLHUP) {
//         closeConn = 1;
//       } 
//       if (res != mscclppSuccess) {
//         WARN("[Proxy Service %d] Failed to execute operation %s from rank %d, retcode %d", comm->rank, mscclppProxyMsgTypeStr[type], comm->localRankToRank[peer->localRank], res);
//         closeConn = 1;
//       }
//       if (closeConn) {
//         mscclppSocketClose(sock);
//         if (op->reqBuff) {
//           free(op->reqBuff);
//           op->reqBuff = NULL;
//         }
//         if (op->respBuff) {
//           free(op->respBuff);
//           op->respBuff = NULL;
//         }
//         op->type = 0;
//         pollfds[s].fd = -1;
//         npeers--;
//       }
//     }
//   }

//   // Wait for all operations to complete and stop progress thread before freeing any resource
//   if (mscclppProxyProgressDestroy(comm) != mscclppSuccess) {
//     WARN("[Proxy Service] proxyDestroy failed");
//   }
//   for (int s=0; s<maxnpeers; s++) {
//     mscclppSocketClose(&peers[s].sock);
//   }
//   mscclppProxyFreeConnections(&connectionPool, comm);
//   mscclppSocketClose(comm->proxyState.listenSock);
//   proxyOpsFree(comm);
//   return NULL;
// }

mscclppResult_t mscclppProxyInit(struct mscclppComm* comm, struct mscclppSocket* sock, union mscclppSocketAddress* peerAddresses) {
  comm->proxyState.listenSock = sock;
  comm->proxyState.peerAddresses = peerAddresses;
  return mscclppSuccess;
}

// mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm) {
//   // comm->proxyState.thread is pthread_join()'d by commFree() in init.cc
//   pthread_create(&comm->proxyState.thread, NULL, mscclppProxyService, comm);
//   mscclppSetThreadName(comm->proxyState.thread, "MSCCLPP Service %2d", comm->cudaDev);
//   return mscclppSuccess;
// }

// mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
//   struct mscclppProxyState* state = &comm->proxyState;

//   if (state == NULL) return mscclppSuccess;
//   if (state->peerAddresses) {
//     if (*comm->abortFlag == 0) {
//       struct mscclppSocket sock;
//       int type = mscclppProxyMsgStop;
//       MSCCLPPCHECK(mscclppSocketInit(&sock, comm->proxyState.peerAddresses + comm->rank, comm->magic, mscclppSocketTypeProxy, comm->abortFlag));
//       MSCCLPPCHECK(mscclppSocketConnect(&sock));
//       MSCCLPPCHECK(mscclppSocketSend(&sock, &type, sizeof(int)));
//       MSCCLPPCHECK(mscclppSocketClose(&sock));
//     }
//     free(state->peerAddresses);
//   }

//   if (state->peerSocks) {
//     for (int i=0; i<comm->localRanks; i++) {
//       int fd;
//       MSCCLPPCHECK(mscclppSocketGetFd(state->peerSocks + i, &fd));
//       if (fd >= 0) {
//         if (state->proxyOps[i].pool) {
//           MSCCLPPCHECK(mscclppShmClose(state->proxyOps[i].handle));
//         }
//         if (state->sharedDevMems[i]) {
//           CUDACHECK(cudaIpcCloseMemHandle(state->sharedDevMems[i]));
//         }
//         int type = mscclppProxyMsgClose;
//         if (*comm->abortFlag == 0) MSCCLPPCHECK(mscclppSocketSend(state->peerSocks + i, &type, sizeof(int)));
//         MSCCLPPCHECK(mscclppSocketClose(state->peerSocks + i));
//       }
//     }
//     free(state->peerSocks);
//     free(state->proxyOps);
//     free(state->sharedDevMems);
//   }
//   return mscclppSuccess;
// }
