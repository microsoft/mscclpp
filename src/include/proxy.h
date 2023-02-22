#ifndef MSCCLPP_PROXY_H_
#define MSCCLPP_PROXY_H_

#include "mscclpp.h"
#include "comm.h"
#include <pthread.h>

struct mscclppProxyState {
  pthread_t *threads;
  int *runs;
};

mscclppResult_t mscclppProxyCreate(struct mscclppComm* comm);
mscclppResult_t mscclppProxyDestroy(struct mscclppComm* comm);

#endif
