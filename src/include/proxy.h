#ifndef MSCCLPP_PROXY_H_
#define MSCCLPP_PROXY_H_

#include "mscclpp.h"
#include "comm.h"
#include <pthread.h>

#define MSCCLPP_PROXY_MAX_NUM (MSCCLPP_IB_MAX_DEVS + 1) // One is for a P2P proxy.

typedef enum {
  MSCCLPP_PROXY_RUN_STATE_IDLE = 0,
  MSCCLPP_PROXY_RUN_STATE_RUNNING,
  MSCCLPP_PROXY_RUN_STATE_EXITING,
} mscclppProxyRunState_t;

struct mscclppProxyState {
  pthread_t thread;
  mscclppProxyRunState_t run;
  mscclppTrigger *cpuTriggerFifo;
  mscclppTrigger *gpuTriggerFifo;
  // cpuTriggerFifoTail indicates where CPU needs to read the head of the fifo.
  unsigned int cpuTriggerFifoTail;
  unsigned int *gpuTriggerFifoHead;
  void *cpuTriggerFifoGdrDesc;
  // NULL for the P2P proxy.
  struct mscclppIbContext *ibContext;
};

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);

#endif
