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
  struct mscclppIbContext* ibCtx;
  cudaStream_t stream;
  struct mscclppProxyState *proxyState;
};

static void readTrigger(mscclppTrigger *dst, mscclppTrigger *src) {
  __m128i xmm0 = _mm_loadu_si128((__m128i *)src);
  _mm_storeu_si128((__m128i *)dst, xmm0);
}

void* mscclppProxyServiceP2P(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  volatile mscclppProxyRunState_t *run = &args->proxyState->run;
  mscclppTrigger *fifo = args->proxyState->cpuTriggerFifo;
  unsigned int *fifoTail = &args->proxyState->cpuTriggerFifoTail;
  cudaStream_t stream = args->stream;
  free(_args);

  // int rank = comm->rank;
  mscclppTrigger trigger;
  // TODO(chhwang): find numa node
  // Current mapping is based on NDv4: GPU [0,1,2,3,4,5,6,7] -> NUMA [1,1,0,0,3,3,2,2]
  // TODO(saemal): either ask user or detect it automatically
  NumaBind((comm->cudaDev / 2) ^ 1);

  PROXYCUDACHECK(cudaSetDevice(comm->cudaDev));
  PROXYCUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  // fifoTail indicates where CPU needs to read the head of the fifo.
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING) break;
    }
    // Poll to see if we are ready to send anything
    readTrigger(&trigger, &fifo[*fifoTail]);
    if (trigger.value[0] == 0) continue;

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
    *(volatile uint64_t *)(&fifo[*fifoTail]) = 0;
    *fifoTail = (*fifoTail + 1) % MSCCLPP_PROXY_FIFO_SIZE;
  }

  // Need a sync in case previous copies are not completed
  PROXYCUDACHECK(cudaStreamSynchronize(stream));

  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;
  PROXYCUDACHECK(cudaStreamDestroy(stream));

  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyServiceIb(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  struct mscclppIbContext *ibCtx = args->ibCtx;
  volatile mscclppProxyRunState_t *run = &args->proxyState->run;
  mscclppTrigger *fifo = args->proxyState->cpuTriggerFifo;
  unsigned int *fifoTail = &args->proxyState->cpuTriggerFifoTail;
  free(_args);

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };
  int *sendState;
  uint64_t *currentProxyFlagVlaue;
  if (mscclppCalloc((void **)&sendState, comm->nConns) != mscclppSuccess) {
    WARN("mscclppCalloc failed: errno %d", errno);
    return NULL;
  }
  if (mscclppCalloc((void **)&currentProxyFlagVlaue, comm->nConns) != mscclppSuccess) {
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
    currentProxyFlagVlaue[i] = *conn->cpuProxyFlag;
    // Post recv
    if (conn->ibQp->postRecv(0) != 0) {
      WARN("postRecv failed: errno %d", errno);
    }
  }
#endif

  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING) break;
    }
    // Poll to see if we are ready to send anything
    readTrigger(&trigger, &fifo[*fifoTail]);

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
          *((volatile uint64_t *)conn->cpuProxyFlag) = ++currentProxyFlagVlaue[trigger.fields.connId];
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
    if (trigger.value[0] == 0) continue;

    struct mscclppConn *conn = &comm->conns[trigger.fields.connId];

    if (trigger.fields.type & mscclppData) {
      conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                            /*wrId=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/false);
    }
    if (trigger.fields.type & mscclppFlag) {
      // My local flag is copied to the peer's proxy flag
      conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                            /*wrId=*/0, /*offset=*/0, /*signaled=*/true);
    }
    int ret;
    if ((ret = conn->ibQp->postSend()) != 0) {
      // Return value is errno.
      WARN("postSend failed: errno %d", ret);
    }

    // Wait for completion
    if (trigger.fields.type & mscclppSync) {
      bool waiting = true;
      while (waiting) {
        wcNum = conn->ibQp->pollCq();
        if (wcNum < 0) {
          WARN("rank %d pollCq failed: errno %d", rank, errno);
          continue;
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
            waiting = false;
            break;
          }
        }
      }
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t *)(&fifo[*fifoTail]) = 0;
    *fifoTail = (*fifoTail + 1) % MSCCLPP_PROXY_FIFO_SIZE;
#endif
  }
  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;
  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyService(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppIbContext *ibCtx = args->ibCtx;
  void *ret;
  if (ibCtx == NULL) {
    ret = mscclppProxyServiceP2P(_args);
  } else {
    ret = mscclppProxyServiceIb(_args);
  }
  return ret;
}

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS + 1; ++i) {
    // `i == MSCCLPP_IB_MAX_DEVS` is for the P2P proxy
    bool is_p2p = (i == MSCCLPP_IB_MAX_DEVS);
    if (!is_p2p) {
      if (comm->ibContext[i] == NULL) continue;
    }
    if (comm->proxyState[i].cpuTriggerFifo == NULL) {
      // reachable when there is no mscclppTransportP2P type connection
      continue;
    }
    struct proxyArgs *args;
    MSCCLPPCHECK(mscclppCalloc(&args, 1));
    args->comm = comm;
    args->ibCtx = is_p2p ? NULL : comm->ibContext[i];
    args->proxyState = &comm->proxyState[i];
    if (is_p2p) {
      CUDACHECK(cudaStreamCreateWithFlags(&args->stream, cudaStreamNonBlocking));
    }
    comm->proxyState[i].run = MSCCLPP_PROXY_RUN_STATE_RUNNING;
    pthread_create(&comm->proxyState[i].thread, NULL, mscclppProxyService, args);
    if (is_p2p) {
      mscclppSetThreadName(comm->proxyState[i].thread, "MSCCLPP Service P2P - %02d", comm->cudaDev);
    } else {
      mscclppSetThreadName(comm->proxyState[i].thread, "MSCCLPP Service IB - %02d", i);
    }
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS + 1; ++i) {
    // `i == MSCCLPP_IB_MAX_DEVS` is for the P2P proxy
    if (i < MSCCLPP_IB_MAX_DEVS) {
      if (comm->ibContext[i] == NULL) continue;
    }
    volatile int *run = (volatile int *)&comm->proxyState[i].run;
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
