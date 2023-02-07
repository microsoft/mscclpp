/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
// #define ENABLE_TIMER 0
// #include "timer.h"

struct mscclppTransport* mscclppTransports[NTRANSPORTS] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport
};

template <int type>
static mscclppResult_t selectTransport(struct mscclppComm* comm, struct mscclppTopoGraph* graph, struct mscclppConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  struct mscclppPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct mscclppPeerInfo* peerInfo = comm->peerInfo+peer;
  struct mscclppConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer].send + connIndex :
                                                  comm->channels[channelId].peers[peer].recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct mscclppTransport *transport = mscclppTransports[t];
    struct mscclppTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    MSCCLPPCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      MSCCLPPCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      return mscclppSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return mscclppSystemError;
}

mscclppResult_t mscclppTransportP2pConnect(struct mscclppComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(MSCCLPP_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct mscclppChannel* channel = &comm->channels[channelId];
  uint64_t mask = 1UL << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].recv[connIndex].connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].send[connIndex].connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return mscclppSuccess;
}

void dumpData(struct mscclppConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct mscclppConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

mscclppResult_t mscclppTransportP2pSetup(struct mscclppComm* comm, struct mscclppTopoGraph* graph, int connIndex, int* highestTransportType/*=NULL*/) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  mscclppResult_t ret = mscclppSuccess;
  int highestType = TRANSPORT_P2P;  // track highest transport type
  struct mscclppConnect data[2*MAXCHANNELS];

//   MSCCLPPCHECKGOTO(mscclppStrongStreamAcquireUncaptured(&comm->hostStream), ret, fail);
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    struct mscclppConnect* recvData = data;
    int sendChannels = 0, recvChannels = 0;
    int type;
    // TIME_START(0);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        MSCCLPPCHECKGOTO(selectTransport<0>(comm, graph, recvData+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
        if (type > highestType) highestType = type;
      }
    }
    // TIME_STOP(0);
    // TIME_START(1);
    struct mscclppConnect* sendData = recvData+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        MSCCLPPCHECKGOTO(selectTransport<1>(comm, graph, sendData+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
        if (type > highestType) highestType = type;
      }
    }
    // TIME_STOP(1);

    // TIME_START(2);
    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         MSCCLPPCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data, sizeof(struct mscclppConnect)*(recvChannels+sendChannels)), ret, fail);
         MSCCLPPCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data, sizeof(struct mscclppConnect)*(recvChannels+sendChannels)), ret, fail);
         sendData = data;
         recvData = data+sendChannels;
      }
    } else {
      if (recvChannels) MSCCLPPCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData, sizeof(struct mscclppConnect)*recvChannels), ret, fail);
      if (sendChannels) MSCCLPPCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData, sizeof(struct mscclppConnect)*sendChannels), ret, fail);
      if (sendChannels) MSCCLPPCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData, sizeof(struct mscclppConnect)*sendChannels), ret, fail);
      if (recvChannels) MSCCLPPCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData, sizeof(struct mscclppConnect)*recvChannels), ret, fail);
    }
    // TIME_STOP(2);

    // TIME_START(3);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        struct mscclppConnector* conn = comm->channels[c].peers[sendPeer].send + connIndex;
        MSCCLPPCHECKGOTO(conn->transportComm->connect(comm, sendData++, 1, comm->rank, conn), ret, fail);
        conn->connected = 1;
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[sendPeer].send[connIndex], &conn->conn, sizeof(struct mscclppConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), ret, fail);
      }
    }
    // TIME_STOP(3);
    // TIME_START(4);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        struct mscclppConnector* conn = comm->channels[c].peers[recvPeer].recv + connIndex;
        MSCCLPPCHECKGOTO(conn->transportComm->connect(comm, recvData++, 1, comm->rank, conn), ret, fail);
        conn->connected = 1;
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[recvPeer].recv[connIndex], &conn->conn, sizeof(struct mscclppConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), ret, fail);
      }
    }
    // TIME_STOP(4);
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

  if (highestTransportType != NULL) *highestTransportType = highestType;
//   TIME_PRINT("P2P Setup/Connect");
exit:
//   MSCCLPPCHECK(mscclppStrongStreamWaitStream(mscclppCudaGraphNone(), &comm->deviceStream, &comm->hostStream));
//   MSCCLPPCHECK(mscclppStrongStreamRelease(mscclppCudaGraphNone(), &comm->hostStream));
  return ret;
fail:
  goto exit;
}

extern struct mscclppTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not MSCCLPPCHECK this call because we would fall back to P2P network in case CollNet setup fails
int mscclppTransportCollNetSetup(struct mscclppComm* comm, struct mscclppTopoGraph* collNetGraph, struct mscclppChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type) {
  int fail = 1;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int rankInCollNet = -1;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    mscclppConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct mscclppPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  // send master receives connect info from peer recv master
  if (isMaster && type == collNetSend) {
    MSCCLPPCHECK(bootstrapRecv(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    TRACE(MSCCLPP_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct mscclppChannelPeer* root = channel->peers+nranks;
  // connector index: 0 for recv, 1 for send
  struct mscclppConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct mscclppTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct mscclppConnect myConnect;
  if (isMaster) {
    MSCCLPPCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  mscclppResult_t res;
  struct {
    int isMaster;
    mscclppConnect connect;
  } *allConnects = NULL;
  mscclppConnect *masterConnects = NULL;
  MSCCLPPCHECK(mscclppCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    MSCCLPPCHECK(mscclppCalloc(&allConnects, nranks));
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct mscclppConnect));
    MSCCLPPCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct mscclppConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct mscclppConnect));
  }
  // connect
  if (isMaster) {
    MSCCLPPCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct mscclppDevChannelPeer* devRoot = channel->devPeers+nranks;
    struct mscclppConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv+type : devRoot->send+type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct mscclppConnInfo), cudaMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == collNetRecv) {
    sendrecvExchange.collNetRank = rankInCollNet;
    memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct mscclppConnect));
    MSCCLPPCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    TRACE(MSCCLPP_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  fail = 0;
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return fail;
}

mscclppResult_t mscclppTransportCollNetCheck(struct mscclppComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[MSCCLPP_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  MSCCLPPCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppTransportCollNetFree(struct mscclppComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct mscclppChannel* channel = comm->channels+r;
    struct mscclppChannelPeer* peer = channel->peers+comm->nRanks;
    for (int b=0; b<MSCCLPP_MAX_CONNS; b++) {
      struct mscclppConnector* send = peer->send + b;
      if (send->transportResources && send->transportComm) MSCCLPPCHECK(send->transportComm->free(send));
      send->transportResources = NULL; // avoid double free
    }
    for (int b=0; b<MSCCLPP_MAX_CONNS; b++) {
      struct mscclppConnector* recv = peer->recv + b;
      if (recv->transportResources && recv->transportComm) MSCCLPPCHECK(recv->transportComm->free(recv));
      recv->transportResources = NULL; // avoid double free
    }
  }
  return mscclppSuccess;
}
