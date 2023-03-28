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

#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

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

void* mscclppProxyServiceP2P(void* _args)
{
  struct proxyArgs* args = (struct proxyArgs*)_args;
  struct mscclppComm* comm = args->comm;
  volatile mscclppProxyRunState_t* run = &args->proxyState->run;
  mscclppTrigger* fifo = args->proxyState->triggerFifo;
  uint64_t* fifoTail = &args->proxyState->fifoTailHost;
  uint64_t fifoTailCached = *fifoTail;
  uint64_t* fifoTailDevPtr = args->proxyState->fifoTailDev;

  cudaStream_t p2pStream = args->proxyState->stream;
  free(_args);

  cudaStream_t stream;
  PROXYCUDACHECK(cudaStreamCreate(&stream));

  // int rank = comm->rank;
  mscclppTrigger trigger;
  // TODO(chhwang): find numa node
  // Current mapping is based on NDv4: GPU [0,1,2,3,4,5,6,7] -> NUMA [1,1,0,0,3,3,2,2]
  // TODO(saemal): either ask user or detect it automatically
  NumaBind((comm->cudaDev / 2) ^ 1);

  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  // fifoTail indicates where CPU needs to read the head of the fifo.
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING)
        break;
    }
    // Poll to see if we are ready to send anything
    readTrigger(&trigger, &fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]);
    if (trigger.value[0] == 0)
      continue; // there is one in progreess
    // there is a trigger value ready to be consumed

    struct mscclppConn* conn = &comm->conns[trigger.fields.connId];

    // Iterate over what send is needed
    if (trigger.fields.type & mscclppData) {
      void* srcBuff = (void*)((char*)conn->devConn->localBuff + trigger.fields.srcDataOffset);
      void* dstBuff = (void*)((char*)conn->devConn->remoteBuff + trigger.fields.dstDataOffset);
      PROXYCUDACHECK(cudaMemcpyAsync(dstBuff, srcBuff, trigger.fields.dataSize, cudaMemcpyDeviceToDevice, p2pStream));

#if defined(ENABLE_NPKIT)
      NpKit::CollectCpuEvent(NPKIT_EVENT_DMA_SEND_ENTRY, (uint32_t)trigger.fields.dataSize,
                             0 /* inflight request differentiator */, *(volatile uint64_t*)NpKit::GetCpuTimestamp(),
                             trigger.fields.connId /* event collection context index */);
#endif
    }
    if (trigger.fields.type & mscclppFlag) {
      PROXYCUDACHECK(cudaMemcpyAsync(conn->remoteProxyFlag, conn->devConn->sendEpochId, sizeof(uint64_t),
                                     cudaMemcpyDeviceToDevice, p2pStream));
    }
    // Wait for completion
    if (trigger.fields.type & mscclppSync) {
      PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
#if defined(ENABLE_NPKIT)
      NpKit::CollectCpuEvent(NPKIT_EVENT_DMA_SEND_EXIT, (uint32_t)trigger.fields.dataSize,
                             0 /* inflight request differentiator */, *(volatile uint64_t*)NpKit::GetCpuTimestamp(),
                             trigger.fields.connId /* event collection context index */);
#endif
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t*)(&fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
    fifoTailCached++;
    if (((fifoTailCached % MSCCLPP_FLUSH_FIFO_COUNTER) == 0) || (trigger.fields.type & mscclppSync))
      PROXYCUDACHECK(cudaMemcpyAsync(fifoTailDevPtr, &fifoTail, sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
  }
  *fifoTail = fifoTailCached;

  // Need a sync in case previous copies are not completed
  PROXYCUDACHECK(cudaStreamSynchronize(p2pStream));
  PROXYCUDACHECK(cudaStreamDestroy(stream));

  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;

  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyServiceIb(void* _args)
{
  struct proxyArgs* args = (struct proxyArgs*)_args;
  struct mscclppComm* comm = args->comm;
  struct mscclppIbContext* ibCtx = args->proxyState->ibContext;
  volatile mscclppProxyRunState_t* run = &args->proxyState->run;
  mscclppTrigger* fifo = args->proxyState->triggerFifo;
  uint64_t* fifoTail = &args->proxyState->fifoTailHost;
  uint64_t fifoTailCached = *fifoTail;
  uint64_t* fifoTailDevPtr = args->proxyState->fifoTailDev;
  free(_args);
  cudaStream_t stream;
  PROXYCUDACHECK(cudaStreamCreate(&stream));

  int rank = comm->rank;
  mscclppTrigger trigger;
  int wcNum;

  NumaBind(ibCtx->numaNode);

  int runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
  for (;;) {
    if (runCheckCounter-- == 0) {
      runCheckCounter = MSCCLPP_PROXY_RUN_STATE_CHECK_PERIOD;
      // Check if we need to exit
      if (*run != MSCCLPP_PROXY_RUN_STATE_RUNNING)
        break;
    }

    // Poll to see if we are ready to send anything
    readTrigger(&trigger, &fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]);
    if (trigger.value[0] == 0)
      continue; // there is one in progreess
    // there is a trigger value ready to be consumed
    struct mscclppConn* conn = &comm->conns[trigger.fields.connId];

    if (trigger.fields.type & mscclppData) {
      conn->ibQp->stageSend(conn->ibBuffMr, &conn->ibBuffMrInfo, (uint32_t)trigger.fields.dataSize,
                            /*wrId=*/0, /*srcOffset=*/trigger.fields.srcDataOffset,
                            /*dstOffset=*/trigger.fields.dstDataOffset,
                            /*signaled=*/false);
#if defined(ENABLE_NPKIT)
      NpKit::CollectCpuEvent(NPKIT_EVENT_IB_SEND_ENTRY, (uint32_t)trigger.fields.dataSize,
                             0 /* inflight request differentiator */, *(volatile uint64_t*)NpKit::GetCpuTimestamp(),
                             trigger.fields.connId /* event collection context index */);
#endif
    }
    if (trigger.fields.type & mscclppFlag) {
      // My local flag is copied to the peer's proxy flag
      conn->ibQp->stageSend(conn->ibLocalFlagMr, &conn->ibProxyFlagMrInfo, sizeof(uint64_t),
                            /*wrId=*/0, /*srcOffset=*/0, /*dstOffset=*/0, /*signaled=*/true);
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
            // send completion
            waiting = false;
#if defined(ENABLE_NPKIT)
            NpKit::CollectCpuEvent(NPKIT_EVENT_IB_SEND_EXIT, (uint32_t)trigger.fields.dataSize,
                                   0 /* inflight request differentiator */,
                                   *(volatile uint64_t*)NpKit::GetCpuTimestamp(),
                                   trigger.fields.connId /* event collection context index */);
#endif
            break;
          }
        }
      }
    }

    // Send completion: reset only the high 64 bits
    *(volatile uint64_t*)(&fifo[fifoTailCached % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
    fifoTailCached++;
    if (((fifoTailCached % MSCCLPP_FLUSH_FIFO_COUNTER) == 0) || (trigger.fields.type & mscclppSync))
      PROXYCUDACHECK(cudaMemcpyAsync(fifoTailDevPtr, &fifoTailCached, sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
  }
  *fifoTail = fifoTailCached;
  PROXYCUDACHECK(cudaStreamDestroy(stream));

  // TODO(saemal): we need to wait for completion of wc here too

  *run = MSCCLPP_PROXY_RUN_STATE_IDLE;
  // WARN("Proxy exits: rank %d", rank);
  return NULL;
}

void* mscclppProxyService(void* _args)
{
  struct proxyArgs* args = (struct proxyArgs*)_args;
  void* ret;
  if (args->proxyState->ibContext == NULL) {
    ret = mscclppProxyServiceP2P(_args);
  } else {
    ret = mscclppProxyServiceIb(_args);
  }
  return ret;
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
