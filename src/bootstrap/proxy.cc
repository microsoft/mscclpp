#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <sys/syscall.h>
#include <map>

#define MSCCLPP_PROXY_FLAG_SET_BY_RDMA 1

struct proxyArgs {
  struct mscclppComm* comm;
  struct mscclppIbContext* ibCtx;
  volatile int* stop;
};

void* mscclppProxyService(void* _args) {
  struct proxyArgs *args = (struct proxyArgs *)_args;
  struct mscclppComm *comm = args->comm;
  struct mscclppIbContext *ibCtx = args->ibCtx;
  volatile int *stop = args->stop;
  free(_args);

  enum {
    SEND_STATE_INIT,
    SEND_STATE_INPROGRESS
  };

  int rank = comm->rank;
  std::map<uint32_t, struct mscclppConn *> qpNumToConn;
  std::map<volatile uint64_t *, std::pair<int, struct mscclppConn *>> trigToSendStateAndConn;
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    if (conn->transport != mscclppTransportIB) continue;
    if (conn->ibCtx != ibCtx) continue;
    volatile uint64_t *tmp = (volatile uint64_t *)conn->devConn->trigger;
    trigToSendStateAndConn[tmp].first = SEND_STATE_INIT;
    trigToSendStateAndConn[tmp].second = conn;
    qpNumToConn[conn->ibQp->qp->qp_num] = conn;
    // All connections may read
    if (conn->ibQp->postRecv(0) != 0) {
      WARN("postRecv failed: errno %d", errno);
    }
  }
  // TODO(chhwang): run send and recv in different threads for lower latency
  mscclppTrigger trigger;
  int wcNum;
  while (*stop == 0) {
    // Try send
    // TODO(chhwang): one thread per conn
    for (auto &pair : trigToSendStateAndConn) {
      if (pair.second.first != SEND_STATE_INIT) continue;
      trigger.value = *pair.first;
      if (trigger.value == 0) continue;
      // Do send
      struct mscclppConn *conn = pair.second.second;
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
      pair.second.first = SEND_STATE_INPROGRESS;
    }

    // Poll completions
    mscclppIbContextPollCq(ibCtx, &wcNum);
    if (wcNum > 0) {
      for (int i = 0; i < wcNum; ++i) {
        struct ibv_wc *wc = &ibCtx->wcs[i];
        if (wc->status != IBV_WC_SUCCESS) {
          WARN("rank %d wc status %d", rank, wc->status);
          continue;
        }
        struct mscclppConn *conn = qpNumToConn[wc->qp_num];
        if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          // recv completion
          if (qpNumToConn[wc->qp_num]->ibQp->postRecv(wc->wr_id) != 0) {
            WARN("postRecv failed: errno %d", errno);
          }
#if (MSCCLPP_PROXY_FLAG_SET_BY_RDMA != 1)
          // TODO(chhwang): gdc & cpu flush
          // *((volatile int *)conn->devConn->remoteFlag) = 1;
#endif
          // WARN("rank %d recv completion", rank);
        } else if (wc->opcode == IBV_WC_RDMA_WRITE) {
          // send completion
          volatile uint64_t *tmp = (volatile uint64_t *)conn->devConn->trigger;
          *tmp = 0;
          trigToSendStateAndConn[tmp].first = SEND_STATE_INIT;
          // WARN("rank %d send completion", rank);
        }
      }
    }
  }
  *stop = 0;
  WARN("Proxy exits: rank %d", rank);
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
    if (comm->ibContext[i] != NULL) {
      struct proxyArgs *args;
      MSCCLPPCHECK(mscclppCalloc(&args, 1));
      args->comm = comm;
      args->ibCtx = comm->ibContext[i];
      args->stop = &comm->proxyState[i].stop;
      pthread_create(&comm->proxyState[i].thread, NULL, mscclppProxyService, args);
      mscclppSetThreadName(comm->proxyState[i].thread, "MSCCLPP Service %2d", i);
    }
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm) {
  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i] != NULL) {
      volatile int *stop = (volatile int *)&comm->proxyState[i].stop;
      *stop = 1;
      while (*stop != 0 && *comm->abortFlag == 0) {
        usleep(1000);
      }
    }
  }
  return mscclppSuccess;
}
