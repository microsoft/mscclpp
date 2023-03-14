/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_INT_NET_H_
#define MSCCLPP_INT_NET_H_

#include "mscclpp.h"
// #include "mscclpp_net.h"
// #include "comm.h"
#include "checks.h"

typedef char mscclppNetHandle_t[MSCCLPP_NET_HANDLE_MAXSIZE];

mscclppResult_t mscclppNetPluginInit();
// mscclppResult_t mscclppNetInit(struct mscclppComm* comm);
// int mscclppNetVersion(struct mscclppComm* comm);

// // Translation to external API
// static const char* mscclppNetName(struct mscclppComm* comm) { return comm->mscclppNet->name; }
// static mscclppResult_t mscclppNetDevices(struct mscclppComm* comm, int* ndev) { MSCCLPPCHECK(comm->mscclppNet->devices(ndev)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetGetProperties(struct mscclppComm* comm, int dev, mscclppNetProperties_t* props) { MSCCLPPCHECK(comm->mscclppNet->getProperties(dev, props)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetListen(struct mscclppComm* comm, int dev, void* handle, void** listenComm) { MSCCLPPCHECK(comm->mscclppNet->listen(dev, handle, listenComm)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetConnect(struct mscclppComm* comm, int dev, void* handle, void** sendComm) { MSCCLPPCHECK(comm->mscclppNet->connect(dev, handle, sendComm)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetAccept(struct mscclppComm* comm, void* listenComm, void** recvComm) { MSCCLPPCHECK(comm->mscclppNet->accept(listenComm, recvComm)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetRegMr(struct mscclppComm* comm, void* netComm, void* data, int size, int type, void** mhandle) { MSCCLPPCHECK(comm->mscclppNet->regMr(netComm, data, size, type, mhandle)); return mscclppSuccess; }
// /* DMA-BUF support */
// static mscclppResult_t mscclppNetRegMrDmaBuf(struct mscclppComm* comm, void* netComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { MSCCLPPCHECK(comm->mscclppNet->regMrDmaBuf(netComm, data, size, type, offset, fd, mhandle)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetDeregMr(struct mscclppComm* comm, void* netComm, void* mhandle) { MSCCLPPCHECK(comm->mscclppNet->deregMr(netComm, mhandle)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetIsend(struct mscclppComm* comm, void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { MSCCLPPCHECK(comm->mscclppNet->isend(sendComm, data, size, tag, mhandle, request)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetIrecv(struct mscclppComm* comm, void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { MSCCLPPCHECK(comm->mscclppNet->irecv(recvComm, n, data, sizes, tags, mhandles, request)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetIflush(struct mscclppComm* comm, void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { MSCCLPPCHECK(comm->mscclppNet->iflush(recvComm, n, data, sizes, mhandles, request)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetTest(struct mscclppComm* comm, void* request, int* done, int* sizes) { MSCCLPPCHECK(comm->mscclppNet->test(request, done, sizes)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetCloseSend(struct mscclppComm* comm, void* sendComm) { MSCCLPPCHECK(comm->mscclppNet->closeSend(sendComm)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetCloseRecv(struct mscclppComm* comm, void* recvComm) { MSCCLPPCHECK(comm->mscclppNet->closeRecv(recvComm)); return mscclppSuccess; }
// static mscclppResult_t mscclppNetCloseListen(struct mscclppComm* comm, void* listenComm) { MSCCLPPCHECK(comm->mscclppNet->closeListen(listenComm)); return mscclppSuccess; }

// // Test whether the current GPU support GPU Direct RDMA.
// mscclppResult_t mscclppGpuGdrSupport(struct mscclppComm* comm, int* gdrSupport);

// extern mscclppNet_t mscclppNetIb;
// extern mscclppNet_t mscclppNetSocket;

#endif
