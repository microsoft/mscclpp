#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ib.h"
#include "socket.h"

#include <emmintrin.h>
#include <map>
#include <numa.h>
#include <sys/syscall.h>
#include <thread>
#include <vector>

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

static void NumaBind(int node)
{
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

struct proxyArgs
{
  struct mscclppComm* comm;
  struct mscclppProxyState* proxyState;
  cudaStream_t stream;
};

static void readTrigger(mscclppTrigger* dst, mscclppTrigger* src)
{
  __m128i xmm0 = _mm_load_si128((__m128i*)src);
  _mm_store_si128((__m128i*)dst, xmm0);
}

#if defined(ENABLE_NPKIT)
static inline void collectNpKitEvent(uint8_t type, uint32_t size, int channelId)
{
  NpKit::CollectCpuEvent(type, size, 0 /* inflight request differentiator */,
                         *(volatile uint64_t*)NpKit::GetCpuTimestamp(), channelId /* event collection context index */);
}
#else
static inline void collectNpKitEvent(uint8_t, uint32_t, int)
{
}
#endif

void* mscclppProxyService(void* _args)
{
  struct proxyArgs* args = (struct proxyArgs*)_args;
  struct mscclppComm* comm = args->comm;
  volatile mscclppProxyRunState_t* run = &args->proxyState->run;
  mscclppTrigger* fifo = args->proxyState->triggerFifo;
  uint64_t* fifoTail = &args->proxyState->fifoTailHost;
  uint64_t* fifoTailDevPtr = args->proxyState->fifoTailDev;
  uint64_t fifoTailCached = *fifoTail;
  mscclppTrigger trigger;
  mscclppIbContext* ibCtx = args->proxyState->ibContext;
  cudaStream_t p2pStream = NULL;
  cudaStream_t stream;

  PROXYCUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  bool isP2pProxy = (ibCtx == nullptr);
  if (isP2pProxy) {
    // TODO(chhwang): find numa node
    // Current mapping is based on NDv4: GPU [0,1,2,3,4,5,6,7] -> NUMA [1,1,0,0,3,3,2,2]
    // TODO(saemal): either ask user or detect it automatically
    NumaBind((comm->cudaDev / 2) ^ 1);
    p2pStream = args->proxyState->stream;
  } else {
    NumaBind(ibCtx->numaNode);
  }
  free(_args); // allocated in mscclppProxyCreate

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
        collectNpKitEvent(NPKIT_EVENT_DMA_SEND_ENTRY, (uint32_t)trigger.fields.dataSize, trigger.fields.connId);
      } else {
        conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                              /*wrId=*/0, /*srcOffset=*/trigger.fields.srcDataOffset,
                              /*dstOffset=*/trigger.fields.dstDataOffset,
                              /*signaled=*/false);
        if ((ret = conn->ibQp->postSend()) != 0) {
          // Return value is errno.
          WARN("data postSend failed: errno %d", ret);
        }
        collectNpKitEvent(NPKIT_EVENT_IB_SEND_ENTRY, (uint32_t)trigger.fields.dataSize, trigger.fields.connId);
      }
    }
    if (trigger.fields.type & mscclppFlag) {
      if (isP2pProxy) {
        PROXYCUDACHECK(cudaMemcpyAsync(conn->remoteProxyFlag, conn->devConn->sendEpochId, sizeof(uint64_t),
                                       cudaMemcpyDeviceToDevice, p2pStream));
      } else {
        // My local flag is copied to the peer's proxy flag
        conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                              /*wrId=*/0, /*srcOffset=*/0, /*dstOffset=*/0, /*signaled=*/true);
        if ((ret = conn->ibQp->postSend()) != 0) {
          WARN("flag postSend failed: errno %d", ret);
        }
      }
    }
    // Wait for completion
    if (trigger.fields.type & mscclppSync) {
      if (isP2pProxy) {
        PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
        collectNpKitEvent(NPKIT_EVENT_DMA_SEND_EXIT, (uint32_t)trigger.fields.dataSize, trigger.fields.connId);
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
        collectNpKitEvent(NPKIT_EVENT_IB_SEND_EXIT, (uint32_t)trigger.fields.dataSize, trigger.fields.connId);
      }
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t*)(&fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
    fifoTailCached++;
    // Flush the tail to device memory. This is either triggered every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER to make sure
    // that the fifo can make progress even if there is no request mscclppSync. However, mscclppSync type is for flush
    // request.
    if (((fifoTailCached % MSCCLPP_PROXY_FIFO_FLUSH_COUNTER) == 0) || (trigger.fields.type & mscclppSync)) {
      PROXYCUDACHECK(
        cudaMemcpyAsync(fifoTailDevPtr, &fifoTailCached, sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }
  }
  *fifoTail = fifoTailCached;

  // make sure the tail is flushed before we shut the proxy
  PROXYCUDACHECK(cudaMemcpyAsync(fifoTailDevPtr, &fifoTailCached, sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
  PROXYCUDACHECK(cudaStreamSynchronize(stream));
  PROXYCUDACHECK(cudaStreamDestroy(stream));
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
