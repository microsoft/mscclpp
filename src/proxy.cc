#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <emmintrin.h>
#include <sys/syscall.h>
#include <numa.h>
#include <map>
#include <vector>
#include <thread>

#define MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD 100
// TODO(chhwang): verify if MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0 is useful, otherwise delete this option.
#define MSCCLPP_PROXY_FLAG_SET_BY_RDMA 1

#define PROXYCUDACHECK(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      WARN("CUDA error from proxy: %s", cudaGetErrorString(err)); \
      return NULL; \
    } \
  } while (false)

static void NumaBind(int node)
{
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

struct proxyArgs {
  struct mscclppComm* comm;
  struct mscclppProxyState *proxyState;
  cudaStream_t stream;
};

static void readTrigger(mscclppTrigger *dst, mscclppTrigger *src) {
  __m128i xmm0 = _mm_load_si128((__m128i *)src);
  _mm_store_si128((__m128i *)dst, xmm0);
}

void* mscclppProxyServiceP2P(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  volatile mscclppProxyRunState_t *run = &args->proxyState->run;
  mscclppTrigger *fifo = args->proxyState->triggerFifo.hostPtr;
  volatile uint64_t *fifoTail = args->proxyState->fifoTail.hostPtr;
  volatile uint64_t *fifoHead = args->proxyState->fifoHead.hostPtr;

  cudaStream_t stream = args->proxyState->stream;
  free(_args);

  // int rank = comm->rank;
  mscclppTrigger trigger;
  // TODO(chhwang): find numa node
  // Current mapping is based on NDv4: GPU [0,1,2,3,4,5,6,7] -> NUMA [1,1,0,0,3,3,2,2]
  // TODO(saemal): either ask user or detect it automatically
  NumaBind((comm->cudaDev / 2) ^ 1);

  uint64_t cachedFifoTail = *fifoTail;
  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  // fifoTail indicates where CPU needs to read the head of the fifo.
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING) break;
    }
    // Poll to see if we are ready to send anything
    if (cachedFifoTail == *fifoHead) continue; // no need trigger
    readTrigger(&trigger, &fifo[cachedFifoTail % MSCCLPP_PROXY_FIFO_SIZE]);
    if (trigger.value[0] == 0) continue; // there is one in progreess
    // there is a trigger value ready to be consumed
    
    struct mscclppConn *conn = &comm->conns[trigger.fields.connId];

    // Iterate over what send is needed
    if (trigger.fields.type & mscclppData){
      void *srcBuff = (void *)((char *)conn->devConn->localBuff + trigger.fields.dataOffset);
      void *dstBuff = (void *)((char *)conn->devConn->remoteBuff + trigger.fields.dataOffset);
      PROXYCUDACHECK(cudaMemcpyAsync(dstBuff, srcBuff, trigger.fields.dataSize, cudaMemcpyDeviceToDevice, stream));
    }
    if (trigger.fields.type & mscclppFlag) {
      PROXYCUDACHECK(cudaMemcpyAsync(conn->remoteProxyFlag, conn->devConn->localFlag, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }
    // Wait for completion
    if (trigger.fields.type & mscclppSync){
      PROXYCUDACHECK(cudaStreamSynchronize(stream));
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t *)(&fifo[cachedFifoTail % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
    cachedFifoTail++;
    *fifoTail = cachedFifoTail;
  }

  // Need a sync in case previous copies are not completed
  PROXYCUDACHECK(cudaStreamSynchronize(stream));

  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;

  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyServiceIb(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  struct mscclppIbContext *ibCtx = args->proxyState->ibContext;
  volatile mscclppProxyRunState_t *run = &args->proxyState->run;
  mscclppTrigger *fifo = args->proxyState->triggerFifo.hostPtr;
  volatile uint64_t *fifoTail = args->proxyState->fifoTail.hostPtr;
  volatile uint64_t *fifoHead = args->proxyState->fifoHead.hostPtr;
  free(_args);

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };
  int *sendState;
  uint64_t *currentProxyFlagValue;
  if (mscclppCalloc((void **)&sendState, comm->nConns) != mscclppSuccess) {
    WARN("mscclppCalloc failed: errno %d", errno);
    return NULL;
  }
  if (mscclppCalloc((void **)&currentProxyFlagValue, comm->nConns) != mscclppSuccess) {
    WARN("mscclppCalloc failed: errno %d", errno);
    return NULL;
  }
#endif

  int rank = comm->rank;
  mscclppTrigger trigger;
  int wcNum;

  NumaBind(ibCtx->numaNode);

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  for (int i = 0; i < (int)comm->nConns; ++i) {
    sendState[i] = SEND_STATE_INIT;
    struct mscclppConn *conn = &comm->conns[i];
    currentProxyFlagValue[i] = *conn->cpuProxyFlag;
    // Post recv
    if (conn->ibQp->postRecv(0) != 0) {
      WARN("postRecv failed: errno %d", errno);
    }
  }
#endif

  std::list<std::pair<uint64_t, struct mscclppConn *>> waitingReqList;
  std::map<struct mscclppConn *, std::list<uint64_t>> waitingReqMap;
  uint64_t cachedFifoTail = *fifoTail;
  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING) break;
    }

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
    struct mscclppConn *conn = &comm->conns[trigger.fields.connId];
    // Try send
    if (sendState[trigger.fields.connId] == SEND_STATE_INIT) {
      if (trigger.value[0] != 0) {
        // Do send
        conn->ibQp->stageSendWithImm(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                                     /*wrId=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/true, /*immData=*/0);
        int ret;
        if ((ret = conn->ibQp->postSend()) != 0) {
          // Return value is errno.
          WARN("postSend failed: errno %d", ret);
        }
        sendState[trigger.fields.connId] = SEND_STATE_INPROGRESS;
      }
    }

    // Poll completions
    wcNum = conn->ibQp->pollCq();
    if (wcNum < 0) {
      WARN("rank %d pollCq failed: errno %d", rank, errno);
    } else {
      for (int i = 0; i < wcNum; ++i) {
        struct ibv_wc *wc = &conn->ibQp->wcs[i];
        if (wc->status != IBV_WC_SUCCESS) {
          WARN("rank %d wc status %d", rank, wc->status);
          continue;
        }
        if (wc->qp_num != conn->ibQp->qp->qp_num) {
          WARN("rank %d got wc of unknown qp_num %d", rank, wc->qp_num);
          continue;
        }
        if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          // TODO(chhwang): cpu flush
          *((volatile uint64_t *)conn->cpuProxyFlag) = ++currentProxyFlagValue[trigger.fields.connId];
          // recv completion
          if (conn->ibQp->postRecv(wc->wr_id) != 0) {
            WARN("postRecv failed: errno %d", errno);
          }
          // WARN("rank %d recv completion", rank);
        } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
          // send completion
          *(volatile uint64_t *)(&fifo[fifoTail]) = 0;
          fifoTail = (fifoTail + 1) % MSCCLPP_PROXY_FIFO_SIZE;
          sendState[trigger.fields.connId] = SEND_STATE_INIT;
          // WARN("rank %d send completion", rank);
        }
      }
    }
#else // (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 1)
    // Poll to see if we are ready to send anything
    if (cachedFifoTail == *fifoHead) continue; // no need trigger
    readTrigger(&trigger, &fifo[cachedFifoTail % MSCCLPP_PROXY_FIFO_SIZE]);
    bool triggered = (trigger.value[0] != 0);
    if (triggered) {
      // there is a trigger value ready to be consumed

      struct mscclppConn *conn = &comm->conns[trigger.fields.connId];

      if (trigger.fields.type & mscclppData) {
        conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                              /*wrId=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/false);
      }
      if (trigger.fields.type & mscclppFlag) {
        // My local flag is copied to the peer's proxy flag
        conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                              /*wrId=*/cachedFifoTail, /*offset=*/0, /*signaled=*/true);
      }
      int ret;
      if ((ret = conn->ibQp->postSend()) != 0) {
        // Return value is errno.
        WARN("postSend failed: errno %d", ret);
      }

      if (trigger.fields.type & mscclppSync) {
        // Reserve waiting for completion
        waitingReqList.emplace_back(cachedFifoTail, conn);
        waitingReqMap[conn].emplace_back(cachedFifoTail);
      } else {
        // No need to wait: reset the trigger immediately
        *(volatile uint64_t *)(&fifo[cachedFifoTail % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
      }
    }

    if (!waitingReqList.empty()) {
      // Need to block until the smallest uncompleted request (waitingReqList.front())
      // is not using the same trigger as the next FIFO tail.
      uint64_t earliestReq = waitingReqList.front().first;
      bool blocking = triggered && (earliestReq == (cachedFifoTail + 1 - MSCCLPP_PROXY_FIFO_SIZE));
      bool blockingDone = false;
      auto it = waitingReqMap.begin();
      for (; it != waitingReqMap.end();) {
        struct mscclppConn *conn = it->first;
        std::list<uint64_t> *reqs = &it->second;
        if (reqs->empty()) {
          ++it;
          continue;
        }
        // If blocking, poll only for the earliest uncompleted request
        if (blocking && (conn != waitingReqList.front().second)) {
          ++it;
          continue;
        }
        // Poll if there is any completion
        wcNum = conn->ibQp->pollCq();
        if (wcNum < 0) {
          WARN("rank %d pollCq failed: errno %d", rank, errno);
          break;
        }
        for (int i = 0; i < wcNum; ++i) {
          struct ibv_wc *wc = &conn->ibQp->wcs[i];
          if (wc->status != IBV_WC_SUCCESS) {
            WARN("rank %d wc status %d", rank, wc->status);
            continue;
          }
          if (wc->qp_num != conn->ibQp->qp->qp_num) {
            WARN("rank %d got wc of unknown qp_num %d", rank, wc->qp_num);
            continue;
          }
          if (wc->opcode == IBV_WC_RDMA_WRITE) {
            // send completion
            *(volatile uint64_t *)(&fifo[wc->wr_id % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
            // reqs is small (at most MSCCLPP_PROXY_FIFO_SIZE), so linear search is fine.
            for (auto it2 = reqs->begin(); it2 != reqs->end(); ++it2) {
              if (*it2 == wc->wr_id) {
                reqs->erase(it2);
                break;
              }
            }
            for (auto it2 = waitingReqList.begin(); it2 != waitingReqList.end(); ++it2) {
              if (it2->first == wc->wr_id) {
                waitingReqList.erase(it2);
                break;
              }
            }
            if (blocking && (wc->wr_id == earliestReq)) {
              // We have completed the earliest uncompleted request, so we can stop polling
              blockingDone = true;
            }
          }
        }
        if (blocking) {
          if (blockingDone) break;
        } else {
          ++it;
        }
      }
    }

    if (triggered) {
      cachedFifoTail++;
      *fifoTail = cachedFifoTail;
    }
#endif
  }

  //TODO(saemal): we need to wait for completion of wc here too

  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;
  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyService(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  void *ret;
  if (args->proxyState->ibContext == NULL) {
    ret = mscclppProxyServiceP2P(_args);
  } else {
    ret = mscclppProxyServiceIb(_args);
  }
  return ret;
}

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState *proxyState = comm->proxyState[i];
    if (proxyState == NULL) break;

    struct proxyArgs *args;
    MSCCLPPCHECK(mscclppCalloc(&args, 1));
    args->comm = comm;
    args->proxyState = proxyState;

    proxyState->run = MSCCLPP_PROXY_RUN_STATE_RUNNING;
    pthread_create(&proxyState->thread, NULL, mscclppProxyService, args);
    if (proxyState->transportType == mscclppTransportP2P) {
      mscclppSetThreadName(proxyState->thread, "MSCCLPP Service P2P - %02d", comm->cudaDev);
    } else if (proxyState->transportType == mscclppTransportIB) {
      mscclppSetThreadName(proxyState->thread, "MSCCLPP Service IB - %02d", i);
    }
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState *proxyState = comm->proxyState[i];
    if (proxyState == NULL) break;

    volatile int *run = (volatile int *)&proxyState->run;
    if (*run == MSCCLPP_PROXY_RUN_STATE_IDLE) {
      continue;
    }
    *run = MSCCLPP_PROXY_RUN_STATE_EXITING;
    while (*run == MSCCLPP_PROXY_RUN_STATE_EXITING && *comm->abortFlag == 0) {
      usleep(1000);
    }
  }
  return mscclppSuccess;
}
