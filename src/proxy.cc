#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ib.h"
#include "socket.h"

#include <emmintrin.h>
#include <map>
#include <sys/syscall.h>
#include <thread>

#include "npkit/npkit.h"

#define MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD 100

#define PROXYCUDACHECK(cmd)                                                                                            \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      WARN("CUDA error from proxy: %s", cudaGetErrorString(err));                                                      \
      return NULL;                                                                                                     \
    }                                                                                                                  \
  } while (false)

#define PROXYMSCCLPPCHECK(call)                                                                                        \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      /* Print the back trace*/                                                                                        \
      if (mscclppDebugNoWarn == 0)                                                                                     \
        INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res);                                                     \
      return NULL;                                                                                                     \
    }                                                                                                                  \
  } while (0);

struct proxyArgs
{
  struct mscclppComm* comm;
  struct mscclppProxyState* proxyState;
};

static void readTrigger(mscclppTrigger* dst, mscclppTrigger* src)
{
  __m128i xmm0 = _mm_load_si128((__m128i*)src);
  _mm_store_si128((__m128i*)dst, xmm0);
}

#if defined(ENABLE_NPKIT)

static void npkitInitReqIds(struct mscclppComm* comm)
{
  for (int i = 0; i < comm->nConns; i++) {
    struct mscclppConn* conn = &comm->conns[i];
    conn->npkitUsedReqIds.resize(0);
    conn->npkitFreeReqIds.resize(MSCCLPP_IB_MAX_SENDS);
    for (uint64_t j = 0; j < MSCCLPP_IB_MAX_SENDS; j++) {
      conn->npkitFreeReqIds[j] = MSCCLPP_IB_MAX_SENDS - j - 1;
    }
  }
}

static void npkitCollectEntryEvent(struct mscclppConn* conn, uint8_t type, uint32_t size, int channelId)
{
  uint64_t reqId = 0;
  if (conn->npkitFreeReqIds.size() == 0) {
    reqId = conn->npkitUsedReqIds.size();
  } else {
    reqId = conn->npkitFreeReqIds.back();
    conn->npkitFreeReqIds.pop_back();
  }
  conn->npkitUsedReqIds.push_back(reqId);
  NpKit::CollectCpuEvent(type, size, (uint32_t)reqId, NpKit::GetCpuTimestamp(), channelId);
}

static void npkitCollectExitEvents(struct mscclppConn* conn, uint8_t type, int channelId)
{
  while (conn->npkitUsedReqIds.size()) {
    uint64_t reqId = conn->npkitUsedReqIds.back();
    NpKit::CollectCpuEvent(type, 0, (uint32_t)reqId, NpKit::GetCpuTimestamp(), channelId);
    conn->npkitFreeReqIds.push_back(reqId);
    conn->npkitUsedReqIds.pop_back();
  }
}

#else

#define npkitInitReqIds(comm)

#define npkitCollectEntryEvent(conn, type, size, channelId)

#define npkitCollectExitEvents(conn, type, channelId)

#endif

void* mscclppProxyService(void* _args)
{
  struct proxyArgs* args = (struct proxyArgs*)_args;
  struct mscclppComm* comm = args->comm;

  // from this point on, proxy thread will stay close to the device
  PROXYCUDACHECK(cudaSetDevice(comm->cudaDev));
  PROXYMSCCLPPCHECK(numaBind(comm->devNumaNode));

  volatile mscclppProxyRunState_t* run = &args->proxyState->run;
  mscclppTrigger* fifo = args->proxyState->triggerFifo;
  uint64_t* fifoTail = &args->proxyState->fifoTailHost;
#if defined(MSCCLPP_USE_GDRCOPY)
  volatile uint64_t* fifoTailDevPtr = args->proxyState->fifoTailDevHostPtr;
#else
  uint64_t* fifoTailDevPtr = args->proxyState->fifoTailDev;
#endif
  uint64_t fifoTailCached = *fifoTail;
  mscclppTrigger trigger;
  mscclppIbContext* ibCtx = args->proxyState->ibContext;
  cudaStream_t p2pStream = args->proxyState->p2pStream;
#if !defined(MSCCLPP_USE_GDRCOPY)
  cudaStream_t fifoStream = args->proxyState->fifoStream;
#endif
  bool isP2pProxy = (ibCtx == nullptr);
  free(_args); // allocated in mscclppProxyCreate

  npkitInitReqIds(comm);

  int counter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  for (;;) {
    if (counter-- == 0) {
      counter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING) {
        break;
      }
    }
    // Poll to see if we are ready to send anything
    readTrigger(&trigger, &fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]);
    if (trigger.value[0] == 0) {
      continue; // there is one in progreess
    }

    struct mscclppConn* conn = &comm->conns[trigger.fields.connId];
    int ret = 0;
    // Iterate over what send is needed
    if (trigger.fields.type & mscclppData) {
      if (isP2pProxy) {
        void* srcBuff = (void*)((char*)conn->devConn->localBuff + trigger.fields.srcDataOffset);
        void* dstBuff = (void*)((char*)conn->devConn->remoteBuff + trigger.fields.dstDataOffset);
        PROXYCUDACHECK(cudaMemcpyAsync(dstBuff, srcBuff, trigger.fields.dataSize, cudaMemcpyDeviceToDevice, p2pStream));
        npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)trigger.fields.dataSize,
                               trigger.fields.connId);
      } else {
        conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                              /*wrId=*/0, /*srcOffset=*/trigger.fields.srcDataOffset,
                              /*dstOffset=*/trigger.fields.dstDataOffset,
                              /*signaled=*/false);
        if ((ret = conn->ibQp->postSend()) != 0) {
          // Return value is errno.
          WARN("data postSend failed: errno %d", ret);
        }
        npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)trigger.fields.dataSize,
                               trigger.fields.connId);
      }
    }
    if (trigger.fields.type & mscclppFlag) {
      if (isP2pProxy) {
        PROXYCUDACHECK(cudaMemcpyAsync(conn->remoteProxyFlag, conn->devConn->sendEpochId, sizeof(uint64_t),
                                       cudaMemcpyDeviceToDevice, p2pStream));
        npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_FLAG_ENTRY, (uint32_t)sizeof(uint64_t),
                               trigger.fields.connId);
      } else {
        // My local flag is copied to the peer's proxy flag
        conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                              /*wrId=*/0, /*srcOffset=*/0, /*dstOffset=*/0, /*signaled=*/true);
        if ((ret = conn->ibQp->postSend()) != 0) {
          WARN("flag postSend failed: errno %d", ret);
        }
        npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_FLAG_ENTRY, (uint32_t)sizeof(uint64_t), trigger.fields.connId);
      }
    }
    // Wait for completion
    if (trigger.fields.type & mscclppSync) {
      if (isP2pProxy) {
        PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
        npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT, trigger.fields.connId);
      } else {
        int rank = comm->rank;
        bool isWaiting = true;
        while (isWaiting) {
          int wcNum = conn->ibQp->pollCq();
          if (wcNum < 0) {
            WARN("rank %d pollCq failed: errno %d", rank, errno);
            continue;
          }
          for (int i = 0; i < wcNum; ++i) {
            struct ibv_wc* wc = &conn->ibQp->wcs[i];
            if (wc->status != IBV_WC_SUCCESS) {
              WARN("rank %d wc status %d", rank, wc->status);
              continue;
            }
            if (wc->qp_num != conn->ibQp->qp->qp_num) {
              WARN("rank %d got wc of unknown qp_num %d", rank, wc->qp_num);
              continue;
            }
            if (wc->opcode == IBV_WC_RDMA_WRITE) {
              isWaiting = false;
              break;
            }
          }
        }
        npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT, trigger.fields.connId);
      }
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t*)(&fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
    fifoTailCached++;
    // Flush the tail to device memory. This is either triggered every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER to make sure
    // that the fifo can make progress even if there is no request mscclppSync. However, mscclppSync type is for flush
    // request.
    if (((fifoTailCached % MSCCLPP_PROXY_FIFO_FLUSH_COUNTER) == 0) || (trigger.fields.type & mscclppSync)) {
#if defined(MSCCLPP_USE_GDRCOPY)
      *fifoTailDevPtr = fifoTailCached;
#else
      PROXYCUDACHECK(
        cudaMemcpyAsync(fifoTailDevPtr, &fifoTailCached, sizeof(uint64_t), cudaMemcpyHostToDevice, fifoStream));
#endif
    }
  }
  *fifoTail = fifoTailCached;

  // make sure the tail is flushed before we shut the proxy
#if defined(MSCCLPP_USE_GDRCOPY)
  *fifoTailDevPtr = fifoTailCached;
#else
  PROXYCUDACHECK(
    cudaMemcpyAsync(fifoTailDevPtr, &fifoTailCached, sizeof(uint64_t), cudaMemcpyHostToDevice, fifoStream));
  PROXYCUDACHECK(cudaStreamSynchronize(fifoStream));
#endif
  if (isP2pProxy) {
    PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
  }
  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;
  return NULL;
}

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm)
{
  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState* proxyState = comm->proxyState[i];
    if (proxyState == NULL)
      break;

    struct proxyArgs* args;
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

mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm)
{
  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState* proxyState = comm->proxyState[i];
    if (proxyState == NULL)
      break;

    volatile int* run = (volatile int*)&proxyState->run;
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
