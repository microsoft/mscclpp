#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <sys/syscall.h>
#include <map>
#include <thread>

#define MSCCLPP_PROXY_FLAG_SET_BY_RDMA 0

struct proxyArgs {
  struct mscclppComm* comm;
  struct mscclppIbContext* ibCtx;
  volatile int* run;
  int connIdx;
};

void* mscclppProxyService(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  volatile int *run = args->run;
  struct mscclppConn *conn = &comm->conns[args->connIdx];
  free(_args);
#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 0)
  int currentRemoteFlagVlaue = *conn->cpuRemoteFlag;
#endif

  // TODO(chhwang): NUMA & core binding

  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };

  int rank = comm->rank;
  int sendState = SEND_STATE_INIT;
  if (conn->ibQp->postRecv(0) != 0) {
    WARN("postRecv failed: errno %d", errno);
  }
  mscclppTrigger trigger;
  int wcNum;
  while (*run) {
    // Try send
    if (sendState == SEND_STATE_INIT) {
      trigger.value = *(volatile uint64_t *)conn->cpuTrigger;
      if (trigger.value != 0) {
        // Do send
#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA == 1)
        conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                              /*wrId=*/0, /*immData=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/false);
        // My local flag is copied to the peer's remote flag
        conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibRemoteFlagMrInfo, sizeof(int),
                              /*wrId=*/0, /*immData=*/0, /*offset=*/0, /*signaled=*/true);
#else
        conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                              /*wrId=*/0, /*immData=*/0, /*offset=*/trigger.fields.dataOffset, /*signaled=*/true);
#endif
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
#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA != 1)
          // TODO(chhwang): cpu flush
          *((volatile int *)conn->cpuRemoteFlag) = ++currentRemoteFlagVlaue;
#endif
          // recv completion
          if (conn->ibQp->postRecv(wc->wr_id) != 0) {
            WARN("postRecv failed: errno %d", errno);
          }
          // WARN("rank %d recv completion", rank);
        } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
          // send completion
          volatile uint64_t *tmp = (volatile uint64_t *)conn->cpuTrigger;
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
  return mscclppSuccess;
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] != NULL) {
      for (int j = 0; j < comm->nConns; ++j) {
        volatile int *run = (volatile int *)&comm->proxyState[i].runs[j];
        if (*run == 0) continue;
        *run = 0;
        while (*run == 0 && *comm->abortFlag == 0) {
          usleep(1000);
        }
        *run = 0;
      }
    }
  }
  return mscclppSuccess;
}
