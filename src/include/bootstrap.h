#pragma once

#include "mscclpp.h"
#include "socket.h"

#include "comm.h"

// ------------------- Old bootstrap headers: to be removed -------------------

struct mscclppBootstrapHandle
{
  uint64_t magic;
  union mscclppSocketAddress addr;
};
mscclppResult_t bootstrapNetInit(const char* ip_port_pair = NULL);
mscclppResult_t bootstrapCreateRoot(struct mscclppBootstrapHandle* handle);
mscclppResult_t bootstrapGetUniqueId(struct mscclppBootstrapHandle* handle, bool isRoot = true,
                                     const char* ip_port_pair = NULL);
mscclppResult_t bootstrapInit(struct mscclppBootstrapHandle* handle, struct mscclppComm* comm);
mscclppResult_t bootstrapAllGather(void* commState, void* allData, int size);
mscclppResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
mscclppResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
mscclppResult_t bootstrapBarrier(void* commState, int* ranks, int rank, int nranks, int tag);
mscclppResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);
mscclppResult_t bootstrapClose(void* commState);
mscclppResult_t bootstrapAbort(void* commState);
