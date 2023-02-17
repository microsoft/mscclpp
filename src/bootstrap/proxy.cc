#include "comm.h"
#include "socket.h"
#include "debug.h"
#include "alloc.h"
#include "ib.h"
#include "checks.h"

#include <sys/syscall.h>
#include <map>

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
  std::map<int, struct mscclppConn *> recvTagToConn;
  std::map<int, struct mscclppConn *> sendTagToConn;
  std::map<struct mscclppConn *, int> sendConnToState;
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    if (conn->transport != mscclppTransportIB) continue;
    if (conn->ibCtx != ibCtx) continue;
    if (conn->rankRecv == rank) {
      recvTagToConn[conn->tag] = conn;
    } else if (conn->rankSend == rank) {
      sendTagToConn[conn->tag] = conn;
      sendConnToState[conn] = SEND_STATE_INIT;
    }
  }
  // Initial post recv
  for (auto &pair : recvTagToConn) {
    struct mscclppConn *conn = pair.second;
    int tag = pair.first;
    if (conn->ibQp->postRecv((uint64_t)-tag) != 0) {
      WARN("postRecv failed: errno %d", errno);
    }
  }
  // TODO(chhwang): run send and recv in different threads for lower latency
  int wcNum;
  while (*stop == 0) {
    // Try send
    for (auto &pair : sendConnToState) {
      if (pair.second == SEND_STATE_INPROGRESS) continue;
      // TODO(chhwang): do we need a thread per flag?
      struct mscclppConn *conn = pair.first;
      volatile int *flag = (volatile int *)conn->flag;
      if (*flag == 0) continue;
      // Do send
      conn->ibQp->stageSend(conn->ibMr, &conn->ibRemoteMrInfo, conn->buffSize,
                            (uint64_t)conn->tag, (unsigned int)conn->tag);
      if (conn->ibQp->postSend() != 0) {
        WARN("postSend failed: errno %d", errno);
      }
      pair.second = SEND_STATE_INPROGRESS;
    }

    // Poll completions
    mscclppIbContextPollCq(ibCtx, &wcNum);
    if (wcNum > 0) {
      for (int i = 0; i < wcNum; ++i) {
        struct ibv_wc *wc = &ibCtx->wcs[i];
        if (wc->status != IBV_WC_SUCCESS) {
          WARN("wc status %d", wc->status);
        }
        if (((int)wc->wr_id) < 0) {
          // recv
          auto search = recvTagToConn.find(wc->imm_data);
          if (search == recvTagToConn.end()) {
            WARN("unexpected imm_data %d", wc->imm_data);
          }
          struct mscclppConn *conn = search->second;
          if (conn->ibQp->postRecv((uint64_t)-wc->imm_data) != 0) {
            WARN("postRecv failed: errno %d", errno);
          }
          volatile int *flag = (volatile int *)conn->flag;
          *flag = 1;
        } else {
          // send
          int tag = (int)wc->wr_id;
          auto search = sendTagToConn.find(tag);
          if (search == sendTagToConn.end()) {
            WARN("unexpected tag %d", tag);
          }
          struct mscclppConn *conn = search->second;
          volatile int *flag = (volatile int *)conn->flag;
          *flag = 0;
          sendConnToState[conn] = SEND_STATE_INIT;
          // WARN("send done rank %d", rank);
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
