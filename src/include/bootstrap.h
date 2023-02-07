/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_BOOTSTRAP_H_
#define MSCCLPP_BOOTSTRAP_H_

#include "mscclpp.h"
#include "socket.h"

#include "comm.h"

struct mscclppBootstrapHandle {
  uint64_t magic;
  union mscclppSocketAddress addr;
};
static_assert(sizeof(struct mscclppBootstrapHandle) <= sizeof(mscclppUniqueId), "Bootstrap handle is too large to fit inside MSCCLPP unique ID");

mscclppResult_t bootstrapNetInit(char* ip_port_pair = NULL);
mscclppResult_t bootstrapCreateRoot(struct mscclppBootstrapHandle* handle, bool idFromEnv);
mscclppResult_t bootstrapGetUniqueId(struct mscclppBootstrapHandle* handle, bool isRoot = true, char* ip_port_pair = NULL);
mscclppResult_t bootstrapInit(struct mscclppBootstrapHandle* handle, struct mscclppComm* comm);
mscclppResult_t bootstrapAllGather(void* commState, void* allData, int size);
mscclppResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
mscclppResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
mscclppResult_t bootstrapBarrier(void* commState, int *ranks, int rank, int nranks, int tag);
mscclppResult_t bootstrapIntraNodeAllGather(void* commState, int *ranks, int rank, int nranks, void* allData, int size);
mscclppResult_t bootstrapClose(void* commState);
mscclppResult_t bootstrapAbort(void* commState);
#endif
