/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_COMM_H_
#define MSCCLPP_COMM_H_

#include "ib.h"
#include "proxy.h"

// #define CACHE_LINE_SIZE 128
// #define MEM_ALIGN 4096
// #define CUDA_IPC_MIN 2097152UL

// // Channels / LL tuning
// #define MSCCLPP_LL_THREAD_THRESHOLD 8
// #define MSCCLPP_LL128_THREAD_THRESHOLD 8
// #define MSCCLPP_SIMPLE_THREAD_THRESHOLD 64

#define MAXCONNECTIONS 64

struct mscclppConn
{
  mscclppTransport_t transport;
  int remoteRank;
  uint64_t buffSize;
  uint64_t* remoteProxyFlag;
  uint64_t* cpuProxyFlag;
  void* cpuProxyFlagGdrDesc;
  struct mscclppDevConn* devConn;
  struct mscclppIbContext* ibCtx;
  struct mscclppIbQp* ibQp;
  struct mscclppIbMr* ibBuffMr;
  struct mscclppIbMr* ibLocalFlagMr;
  struct mscclppIbMr* ibProxyFlagMr;
  struct mscclppIbMrInfo ibBuffMrInfo;
  struct mscclppIbMrInfo ibLocalFlagMrInfo;
  struct mscclppIbMrInfo ibProxyFlagMrInfo;
};

struct mscclppComm
{
  struct mscclppConn conns[MAXCONNECTIONS];
  struct mscclppDevConn devConns[MAXCONNECTIONS];
  int nConns;

  void* bootstrap;

  uint64_t
    magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  // Flag to ask MSCCLPP kernels to abort
  volatile uint32_t* abortFlag;

  struct mscclppIbContext* ibContext[MSCCLPP_IB_MAX_DEVS];
  cudaStream_t stream; // DMA engine stream for P2P
  struct mscclppProxyState* proxyState[MSCCLPP_PROXY_MAX_NUM];
};

#endif
