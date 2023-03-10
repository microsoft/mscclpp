#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <sys/syscall.h>
#include <numa.h>
#include <map>
#include <vector>
#include <thread>

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
  volatile mscclppProxyRunState_t* run;
  int connIdx;
};

// TODO(saemal) We need to add a fifo for each DMA engine
void* mscclppProxyServiceP2P(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  volatile mscclppProxyRunState_t *run = args->run;
  std::vector<struct mscclppConn *> conns;
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    // TODO(saemal): we need to create another transport type which doesn't need a proxy.
    if (conn->transport == mscclppTransportP2P) {
      conns.push_back(conn);
    }
  }
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

  while (*run == MSCCLPP_PROXY_RUN_STATE_RUNNING) {
    for (struct mscclppConn *conn : conns) {
      // Poll to see if we are ready to send anything
      trigger.value[0] = *(volatile uint64_t *)(conn->cpuTriggerFifo[conn->fifoTail].value);
      if (trigger.value[0] == 0) continue;
      // TODO(chhwang): latency overhead of reading value[1] is too large (~9us)
      trigger.value[1] = *(volatile uint64_t *)(conn->cpuTriggerFifo[conn->fifoTail].value + 1);
      if (trigger.value[1] != 42) {
        WARN("Unexpected value");
      }

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
      volatile uint64_t *tmp = (volatile uint64_t *)(&conn->cpuTriggerFifo[conn->fifoTail]);
      *tmp = 0;
      conn->fifoTail++;
      if (conn->fifoTail == MSCCLPP_PROXY_FIFO_SIZE)
        conn->fifoTail = 0;
    }
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
  volatile mscclppProxyRunState_t *run = args->run;
  std::vector<struct mscclppConn *> conns;
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    if (conn->transport == mscclppTransportIB) {
      conns.push_back(conn);
    }
  }
  free(_args);

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };
  int sendState = SEND_STATE_INIT;
  uint64_t currentProxyFlagVlaue = *conn->cpuProxyFlag;
#endif

  int rank = comm->rank;
  mscclppTrigger trigger;
  int wcNum;

  NumaBind(ibCtx->numaNode);

#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  for (struct mscclppConn *conn : conns) {
    // Post recv
    if (conn->ibQp->postRecv(0) != 0) {
      WARN("postRecv failed: errno %d", errno);
    }
  }
#endif

  while (*run == MSCCLPP_PROXY_RUN_STATE_RUNNING) {
    for (struct mscclppConn *conn : conns) {
#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
      // Try send
      if (sendState == SEND_STATE_INIT) {
        trigger.value = *(volatile uint64_t *)(&conn->cpuTriggerFifo[conn->fifoTail]);
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
            volatile uint64_t *tmp = (volatile uint64_t *)(&conn->cpuTriggerFifo[conn->fifoTail]);
            *tmp = 0;
            conn->fifoTail++;
            if (conn->fifoTail == MSCCLPP_PROXY_FIFO_SIZE)
              conn->fifoTail = 0;
            sendState = SEND_STATE_INIT;
            // WARN("rank %d send completion", rank);
          }
        }
      }
#else // (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 1)
      // Poll to see if we are ready to send anything
      trigger.value[0] = *(volatile uint64_t *)(conn->cpuTriggerFifo[conn->fifoTail].value);
      if (trigger.value[0] == 0) continue;
      // TODO(chhwang): latency overhead of reading value[1] is too large (~9us)
      trigger.value[1] = *(volatile uint64_t *)(conn->cpuTriggerFifo[conn->fifoTail].value + 1);
      if (trigger.value[1] != 42) {
        WARN("Unexpected value");
      }

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

      // Send completion: reset only the high 64 bits
      volatile uint64_t *tmp = (volatile uint64_t *)(&conn->cpuTriggerFifo[conn->fifoTail]);
      *tmp = 0;
      conn->fifoTail++;
      if (conn->fifoTail == MSCCLPP_PROXY_FIFO_SIZE)
        conn->fifoTail = 0;
#endif
    }
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

// mscclppResult_t mscclppProxyInit(struct mscclppComm* comm, struct mscclppSocket* sock, union mscclppSocketAddress* peerAddresses) {
//   comm->proxyState.listenSock = sock;
//   comm->proxyState.peerAddresses = peerAddresses;
//   return mscclppSuccess;
// }

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] == NULL) continue;
    if (comm->proxyState[i].threads == NULL) {
      MSCCLPPCHECK(mscclppCalloc(&comm->proxyState[i].threads, 1));
    }
    if (comm->proxyState[i].runs == NULL) {
      MSCCLPPCHECK(mscclppCalloc(&comm->proxyState[i].runs, 1));
    }
    // Create IB proxy threads
    struct proxyArgs *args;
    MSCCLPPCHECK(mscclppCalloc(&args, 1));
    args->comm = comm;
    args->ibCtx = comm->ibContext[i];
    args->run = comm->proxyState[i].runs;
    *args->run = MSCCLPP_PROXY_RUN_STATE_RUNNING;
    pthread_create(comm->proxyState[i].threads, NULL, mscclppProxyService, args);
    mscclppSetThreadName(comm->proxyState[i].threads[0], "MSCCLPP Service IB - %02d", i);
  }
  // P2P proxy
  mscclppProxyState *proxyState = &comm->proxyState[MSCCLPP_IB_MAX_DEVS];
  if (proxyState->threads == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&proxyState->threads, 1));
  }
  if (proxyState->runs == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&proxyState->runs, 1));
  }
  // Create P2P DMA proxy thread
  struct proxyArgs *args;
  MSCCLPPCHECK(mscclppCalloc(&args, 1));
  args->comm = comm;
  args->ibCtx = NULL;
  args->run = proxyState->runs;
  args->connIdx = -1; // unused
  CUDACHECK(cudaStreamCreateWithFlags(&args->stream, cudaStreamNonBlocking));
  *args->run = MSCCLPP_PROXY_RUN_STATE_RUNNING;
  pthread_create(proxyState->threads, NULL, mscclppProxyService, args);
  mscclppSetThreadName(proxyState->threads[0], "MSCCLPP Service P2P - %02d", comm->cudaDev);
  return mscclppSuccess;
}

static void _stopProxy(struct mscclppComm* comm, int devIdx, int connIdx) {
  volatile int *run = (volatile int *)&comm->proxyState[devIdx].runs[connIdx];
  if (*run == MSCCLPP_PROXY_RUN_STATE_IDLE) return;
  *run = MSCCLPP_PROXY_RUN_STATE_EXITING;
  while (*run == MSCCLPP_PROXY_RUN_STATE_EXITING && *comm->abortFlag == 0) {
    usleep(1000);
  }
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] != NULL) {
      _stopProxy(comm, i, 0);
    }
  }
  // P2P proxies
  _stopProxy(comm, MSCCLPP_IB_MAX_DEVS, 0);
  return mscclppSuccess;
}
