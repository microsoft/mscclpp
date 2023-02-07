/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_TRANSPORT_H_
#define MSCCLPP_TRANSPORT_H_

#include "devcomm.h"
#include "graph.h"
// #include "nvmlwrap.h"
#include "core.h"

#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"

extern struct mscclppTransport p2pTransport;
extern struct mscclppTransport shmTransport;
extern struct mscclppTransport netTransport;
extern struct mscclppTransport collNetTransport;

extern struct mscclppTransport* mscclppTransports[];

// Forward declarations
struct mscclppRing;
struct mscclppConnector;
struct mscclppComm;

struct mscclppPeerInfo {
  int rank;
  int cudaDev;
  int netDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  struct mscclppComm* comm;
  int cudaCompCap;
};

#define CONNECT_SIZE 128
struct mscclppConnect {
  char data[CONNECT_SIZE];
};

struct mscclppTransportComm {
  mscclppResult_t (*setup)(struct mscclppComm* comm, struct mscclppTopoGraph* graph, struct mscclppPeerInfo*, struct mscclppPeerInfo*, struct mscclppConnect*, struct mscclppConnector*, int channelId, int connIndex);
  mscclppResult_t (*connect)(struct mscclppComm* comm, struct mscclppConnect*, int nranks, int rank, struct mscclppConnector*);
  mscclppResult_t (*free)(struct mscclppConnector*);
  mscclppResult_t (*proxySharedInit)(struct mscclppProxyConnection* connection, struct mscclppComm* comm, int nChannels);
  mscclppResult_t (*proxySetup)(struct mscclppProxyConnection* connection, struct mscclppComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  mscclppResult_t (*proxyConnect)(struct mscclppProxyConnection* connection, struct mscclppComm* comm, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  mscclppResult_t (*proxyFree)(struct mscclppProxyConnection* connection, struct mscclppComm* comm);
  mscclppResult_t (*proxyProgress)(struct mscclppComm* comm, struct mscclppProxyArgs*);
};

struct mscclppTransport {
  const char name[4];
  mscclppResult_t (*canConnect)(int*, struct mscclppTopoSystem* topo, struct mscclppTopoGraph* graph, struct mscclppPeerInfo*, struct mscclppPeerInfo*);
  struct mscclppTransportComm send;
  struct mscclppTransportComm recv;
};

mscclppResult_t mscclppTransportP2pConnect(struct mscclppComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
mscclppResult_t mscclppTransportP2pSetup(struct mscclppComm* comm, struct mscclppTopoGraph* graph, int connIndex, int* highestTransportType=NULL);

enum { collNetRecv=0, collNetSend=1 };
int mscclppTransportCollNetSetup(struct mscclppComm* comm, struct mscclppTopoGraph* collNetGraph, struct mscclppChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type);
mscclppResult_t mscclppTransportCollNetCheck(struct mscclppComm* comm, int collNetSetupFail);
mscclppResult_t mscclppTransportCollNetFree(struct mscclppComm* comm);
#endif
