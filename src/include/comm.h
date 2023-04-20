/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_COMM_H_
#define MSCCLPP_COMM_H_

#include "ib.h"
#include "proxy.h"
#include <vector>

#define MAXCONNECTIONS 64

struct mscclppBufferRegistration
{
  void *data;
  uint64_t size;
};

struct mscclppConn
{
  int connId;
  mscclppTransport_t transport;
  int remoteRank;
  uint64_t buffSize;
  struct mscclppDevConn* devConn;
  struct mscclppHostConn* hostConn;

  std::vector<mscclppBufferRegistration> bufferRegistrations;
  std::vector<mscclppBufferRegistration> remoteBufferRegistrations;

  struct mscclppIbContext* ibCtx;
#if defined(ENABLE_NPKIT)
  std::vector<uint64_t> npkitUsedReqIds;
  std::vector<uint64_t> npkitFreeReqIds;
#endif
};

struct mscclppComm
{
  struct mscclppConn conns[MAXCONNECTIONS];
  struct mscclppDevConn devConns[MAXCONNECTIONS];
  int nConns;

  MscclppBootstrap bootstrap;

  // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.
  uint64_t magic;

  int rank;        // my rank in the communicator
  int nRanks;      // number of GPUs in communicator
  int cudaDev;     // my cuda device index
  int devNumaNode; // my device's NUMA node

  // Flag to ask MSCCLPP kernels to abort
  volatile uint32_t* abortFlag;

  struct mscclppIbContext* ibContext[MSCCLPP_IB_MAX_DEVS];
  struct mscclppProxyState* proxyState[MSCCLPP_PROXY_MAX_NUM];
};

#endif
