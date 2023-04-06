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

struct mscclppProxyState
{
  mscclppTransport_t transportType;
  pthread_t thread;
  mscclppProxyRunState_t run;

  // fifo cudaHostCalloc'ed that is produced by device and consumed by host
  mscclppTrigger* triggerFifo;
#if defined(MSCCLPP_USE_GDR)
  mscclppTrigger* triggerFifoDev;
  void* triggerFifoDesc;
#endif
  // allocated on the device and only accessed by the device
  uint64_t* fifoHead;

  // allocated on the device. Read-only by device, write-only by host
  uint64_t* fifoTailDev;
  // allocated on the host. Only accessed by the host. This is a copy of the
  // value pointed to by fifoTailDev and the invariant is that
  // *fifoTailDev <= fifoTailHost. Meaning that host's copy of tail is
  // always ahead of the device's copy and host updates the device's copy
  // only when it is needed. Therefore, fifoTailHost is the "true" tail
  // and fifoTailDev is a "stale" tail. See proxy.cc to undertand how
  // these updates are pushed to the device.
  uint64_t fifoTailHost;

  int numaNodeToBind;
  struct mscclppIbContext* ibContext; // For IB connection only
  cudaStream_t p2pStream;             // for P2P DMA engine only
  cudaStream_t fifoStream;            // for transferring fifo tail
};

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);

#endif
