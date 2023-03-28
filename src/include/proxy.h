#ifndef MSCCLPP_PROXY_H_
#define MSCCLPP_PROXY_H_

#include "comm.h"
#include "mscclpp.h"
#include <cuda_runtime.h>
#include <pthread.h>

#define MSCCLPP_PROXY_MAX_NUM (MSCCLPP_IB_MAX_DEVS + 1) // One is for a P2P proxy.

typedef enum
{
  MSCCLPP_PROXY_RUN_STATE_IDLE = 0,
  MSCCLPP_PROXY_RUN_STATE_RUNNING,
  MSCCLPP_PROXY_RUN_STATE_EXITING,
} mscclppProxyRunState_t;

template <typename T> struct mscclppGDRState
{
  T* hostPtr;
  T* devPtr;
  void* desc;
};

struct mscclppProxyState
{
  mscclppTransport_t transportType;
  pthread_t thread;
  mscclppProxyRunState_t run;

  // fifo allocation that is accessible on both host and device
  mscclppGDRState<mscclppTrigger> triggerFifo;
  mscclppGDRState<uint64_t> fifoHead;
  mscclppGDRState<uint64_t> fifoTail;

  struct mscclppIbContext* ibContext; // For IB connection only
  cudaStream_t stream;                // for P2P DMA engine only
};

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);

#endif
