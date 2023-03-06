#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <sys/syscall.h>
#include <numa.h>
#include <map>
#include <thread>

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
  volatile int* run;
  int connIdx;
};

// TODO(saemal) We need to add a fifo for each DMA engine
void* mscclppProxyServiceP2P(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  // TODO(saemal): we perhaps need a finite state for run instead of just 0 and 1
  volatile int *run = args->run;
  struct mscclppConn *conn = &comm->conns[args->connIdx];
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

  while (*run) {
    // Poll to see if we are ready to send anything
    trigger.value = *(volatile uint64_t *)(&conn->cpuTriggerFifo[conn->fifoTail]);
    if (trigger.value == 0) continue;

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

    // send completion
    volatile uint64_t *tmp = (volatile uint64_t *)conn->cpuTriggerFifo;
    *tmp = 0;
    conn->fifoTail++;
  }
  *run = 1;
  PROXYCUDACHECK(cudaStreamDestroy(stream));

  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)

// TODO(saemal) We need to add a fifo for each DMA engine
void* mscclppProxyServiceIb(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  struct mscclppIbContext *ibCtx = args->ibCtx;
  volatile int *run = args->run;
  struct mscclppConn *conn = &comm->conns[args->connIdx];
  free(_args);
  uint64_t currentProxyFlagVlaue = *conn->cpuProxyFlag;

  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };

  int rank = comm->rank;
  int sendState = SEND_STATE_INIT;
  mscclppTrigger trigger;
  int wcNum;

  NumaBind(ibCtx->numaNode);
  if (conn->ibQp->postRecv(0) != 0) {
    WARN("postRecv failed: errno %d", errno);
  }

  while (*run) {
    // Try send
    if (sendState == SEND_STATE_INIT) {
      trigger.value = *(volatile uint64_t *)conn->cpuTriggerFifo;
      if (trigger.value != 0) {
        // Do send
        conn->ibQp->stageSendWithImm(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                                     /*wrId=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/true, /*immData=*/0);
        if (conn->ibQp->postSend() != 0) {
          WARN("postSend failed: errno %d", errno);
        }
        sendState = SEND_STATE_INPROGRESS;
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
          *((volatile uint64_t *)conn->cpuProxyFlag) = ++currentProxyFlagVlaue;
          // recv completion
          if (conn->ibQp->postRecv(wc->wr_id) != 0) {
            WARN("postRecv failed: errno %d", errno);
          }
          // WARN("rank %d recv completion", rank);
        } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
          // send completion
          volatile uint64_t *tmp = (volatile uint64_t *)conn->cpuTriggerFifo;
          *tmp = 0;
          sendState = SEND_STATE_INIT;
          // WARN("rank %d send completion", rank);
        }
      }
    }
  }
  *run = 1;
  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

#else // MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 1

// TODO(saemal): merge this with the function above
void* mscclppProxyServiceIb(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  struct mscclppIbContext *ibCtx = args->ibCtx;
  volatile int *run = args->run;
  struct mscclppConn *conn = &comm->conns[args->connIdx];
  free(_args);

  int rank = comm->rank;
  mscclppTrigger trigger;
  int wcNum;

  NumaBind(ibCtx->numaNode);

  while (*run) {
    // Poll to see if we are ready to send anything
    trigger.value = *(volatile uint64_t *)conn->cpuTriggerFifo;
    if (trigger.value == 0) continue;

    if (trigger.fields.type & mscclppData) {
      conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                            /*wrId=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/false);
    }
    if (trigger.fields.type & mscclppFlag) {
        // My local flag is copied to the peer's proxy flag
        conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                              /*wrId=*/0, /*offset=*/0, /*signaled=*/true);
    }
    if (conn->ibQp->postSend() != 0) {
      WARN("postSend failed: errno %d", errno);
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

    // Send completion
    volatile uint64_t *tmp = (volatile uint64_t *)conn->cpuTriggerFifo;
    *tmp = 0;
  }
  *run = 1;
  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

#endif

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

// mscclppResult_t mscclppProxyInit(struct mscclppComm* comm, struct mscclppSocket* sock, union mscclppSocketAddress* peerAddresses) {
//   comm->proxyState.listenSock = sock;
//   comm->proxyState.peerAddresses = peerAddresses;
//   return mscclppSuccess;
// }

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm) {
  // comm->proxyState.thread is pthread_join()'d by commFree() in init.cc
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] == NULL) continue;
    if (comm->proxyState[i].threads == NULL) {
      MSCCLPPCHECK(mscclppCalloc(&comm->proxyState[i].threads, comm->nConns));
    }
    if (comm->proxyState[i].runs == NULL) {
      MSCCLPPCHECK(mscclppCalloc(&comm->proxyState[i].runs, comm->nConns));
    }
    for (int j = 0; j < comm->nConns; ++j) {
      // Create IB proxy threads
      struct mscclppConn *conn = &comm->conns[j];
      if (conn->transport != mscclppTransportIB) continue;
      if (conn->ibCtx != comm->ibContext[i]) continue;
      struct proxyArgs *args;
      MSCCLPPCHECK(mscclppCalloc(&args, 1));
      args->comm = comm;
      args->ibCtx = comm->ibContext[i];
      args->run = &comm->proxyState[i].runs[j];
      args->connIdx = j;
      *args->run = 1;
      pthread_create(&comm->proxyState[i].threads[j], NULL, mscclppProxyService, args);
      mscclppSetThreadName(comm->proxyState[i].threads[j], "MSCCLPP Service %2d - %4d", i, j);
    }
  }
  // P2P proxies
  mscclppProxyState *proxyState = &comm->proxyState[MSCCLPP_IB_MAX_DEVS];
  if (proxyState->threads == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&proxyState->threads, comm->nConns));
  }
  if (proxyState->runs == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&proxyState->runs, comm->nConns));
  }
  for (int j = 0; j < comm->nConns; ++j) {
    // Create P2P DMA proxy threads
    if (comm->conns[j].transport != mscclppTransportP2P) continue;
    struct proxyArgs *args;
    MSCCLPPCHECK(mscclppCalloc(&args, 1));
    args->comm = comm;
    args->ibCtx = NULL;
    args->run = &proxyState->runs[j];
    args->connIdx = j;
    CUDACHECK(cudaStreamCreateWithFlags(&args->stream, cudaStreamNonBlocking));
    *args->run = 1;
    pthread_create(&proxyState->threads[j], NULL, mscclppProxyService, args);
    mscclppSetThreadName(proxyState->threads[j], "MSCCLPP Service %2d - %4d", MSCCLPP_IB_MAX_DEVS + 1, j);
  }
  return mscclppSuccess;
}

static void _stopProxy(struct mscclppComm* comm, int devIdx, int connIdx) {
  volatile int *run = (volatile int *)&comm->proxyState[devIdx].runs[connIdx];
  if (*run == 0) return;
  *run = 0;
  while (*run == 0 && *comm->abortFlag == 0) {
    usleep(1000);
  }
  *run = 0;
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] != NULL) {
      for (int j = 0; j < comm->nConns; ++j) {
        _stopProxy(comm, i, j);
      }
    }
  }
  // P2P proxies
  mscclppProxyState *proxyState = &comm->proxyState[MSCCLPP_IB_MAX_DEVS];
  for (int j = 0; j < comm->nConns; ++j) {
    if (comm->conns[j].transport != mscclppTransportP2P) continue;
    _stopProxy(comm, MSCCLPP_IB_MAX_DEVS, j);
  }
  return mscclppSuccess;
}
