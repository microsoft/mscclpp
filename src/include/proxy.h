#ifndef MSCCLPP_PROXY_H_
#define MSCCLPP_PROXY_H_

#include "mscclpp.h"
#include "comm.h"
#include <pthread.h>

typedef enum {
  MSCCLPP_PROXY_RUN_STATE_IDLE = 0,
  MSCCLPP_PROXY_RUN_STATE_RUNNING,
  MSCCLPP_PROXY_RUN_STATE_EXITING,
} mscclppProxyRunState_t;

struct mscclppProxyState {
  pthread_t *threads;
  mscclppProxyRunState_t *runs;
};

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);

#endif
