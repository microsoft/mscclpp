/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "bootstrap.h"
#include "config.h"
#include "mscclpp.h"
#include "utils.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>

#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

namespace {
uint64_t hashUniqueId(const mscclppBootstrapHandle& id)
{
  const char* bytes = (const char*)&id;
  uint64_t h = 0xdeadbeef;
  for (int i = 0; i < (int)sizeof(mscclppBootstrapHandle); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

mscclppResult_t setFilesLimit()
{
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return mscclppSuccess;
}

} // namespace

/* Socket Interface Selection type */
enum bootstrapInterface_t
{
  findSubnetIf = -1,
  dontCareIf = -2
};

struct MscclppBootstrap::UniqueId
{
  uint64_t magic;
  union mscclppSocketAddress addr;
};

struct unexpectedConn
{
  int peer;
  int tag;
  struct mscclppSocket sock;
};

struct extInfo
{
  int rank;
  int nRanks;
  union mscclppSocketAddress extAddressListenRoot;
  union mscclppSocketAddress extAddressListen;
};

class MscclppBootstrap::Impl
{
public:
  static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
  static union mscclppSocketAddress bootstrapNetIfAddr;

  static void bootstrapRoot(mscclppSocket* listenSock, uint64_t magic, int nRanks);

  Impl(std::string ipPortPair, int rank, int nRanks, const mscclppBootstrapHandle handle);
  mscclppResult_t init(const mscclppComm& comm);
  mscclppResult_t createRoot(MscclppBootstrap::UniqueId& handle);
  mscclppResult_t allGather(void* allData, int size);

  void startBootstrapThread();

  MscclppBootstrap::UniqueId uniqueId_;
private:
  int rank_;
  int nRanks_;
  mscclppSocket listenSock_;
  mscclppSocket ringRecvSocket_;
  mscclppSocket ringSendSocket_;
  std::vector<mscclppSocketAddress> peerCommAddresses_;
  std::vector<mscclppSocketAddress> peerProxyAddresses_;
  std::queue<unexpectedConn> unexpectedConnections_;
  volatile uint32_t* abortFlag_;

  static mscclppResult_t netSend(mscclppSocket* sock, void* data, int size);
  static mscclppResult_t netRecv(mscclppSocket* sock, void* data, int size);

  mscclppResult_t netInit(std::string ipPortPair);
};

MscclppBootstrap::Impl::Impl(std::string ipPortPair, int rank, int nRanks, const mscclppBootstrapHandle handle)
  : rank_(rank), nRanks_(nRanks), peerCommAddresses_(nRanks, mscclppSocketAddress()),
    peerProxyAddresses_(nRanks, mscclppSocketAddress()), abortFlag_(nullptr)
{
  int ret = netInit(ipPortPair);
  if (ret != mscclppSuccess) {
    throw std::runtime_error("Failed to initialize network");
  }

  mscclppBootstrapHandle zeroHandle = {0};
  if (memcmp(&handle, &zeroHandle, sizeof(mscclppBootstrapHandle)) != 0) {
    uniqueId_.magic = handle.magic;
    uniqueId_.addr = handle.addr;
    return;
  }

  mscclppResult_t ret = getRandomData(&uniqueId_.magic, sizeof(uniqueId_.magic));
  if (ret != mscclppSuccess) {
    throw std::runtime_error("getting random data failed");
  }
  std::memcpy(&uniqueId_.addr, &bootstrapNetIfAddr, sizeof(union mscclppSocketAddress));
}

mscclppResult_t MscclppBootstrap::Impl::netInit(std::string ipPortPair)
{
  if (!ipPortPair.empty()) {
    union mscclppSocketAddress remoteAddr;
    if (mscclppSocketGetAddrFromString(&remoteAddr, ipPortPair.c_str()) != mscclppSuccess) {
      WARN("Invalid MSCCLPP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return mscclppInvalidArgument;
    }
    if (mscclppFindInterfaceMatchSubnet(this->bootstrapNetIfName, &this->bootstrapNetIfAddr, &remoteAddr,
                                        MAX_IF_NAME_SIZE, 1) <= 0) {
      WARN("NET/Socket : No usable listening interface found");
      return mscclppSystemError;
    }
  } else {
    int ret = mscclppFindInterfaces(this->bootstrapNetIfName, &this->bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      WARN("Bootstrap : no socket interface found");
      return mscclppInternalError;
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", bootstrapNetIfName);
  mscclppSocketToString(&bootstrapNetIfAddr, line + strlen(line));
  INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
  return mscclppSuccess;
}


MscclppBootstrap::MscclppBootstrap(std::string ipPortPair, int rank, int nRanks)
{
  pimpl = std::make_unique<Impl>(ipPortPair, rank, nRanks, mscclppBootstrapHandle{0});
}

MscclppBootstrap::MscclppBootstrap(mscclppBootstrapHandle handle, int rank, int nRanks)
{
  pimpl = std::make_unique<Impl>("", rank, nRanks, handle);
}

MscclppBootstrap::UniqueId MscclppBootstrap::getUniqueId()
{
  return pimpl->uniqueId_;
}

// void MscclppBootstrap::Impl::bootstrapRoot(mscclppSocket* listenSock, uint64_t magic, int nRanks)
// {
//   extInfo info;
//   mscclppResult_t res = mscclppSuccess;
//   int numCollected = 0;
//   std::vector<mscclppSocketAddress> rankAddresses(nRanks, mscclppSocketAddress());
//   // for initial rank <-> root information exchange
//   std::vector<mscclppSocketAddress> rankAddressesRoot(nRanks, mscclppSocketAddress());

//   mscclppSocketAddress zero;
//   std::memset(rankAddresses.data(), 0, sizeof(mscclppSocketAddress) * nRanks);
//   std::memset(rankAddressesRoot.data(), 0, sizeof(mscclppSocketAddress) * nRanks);
//   std::memset(&zero, 0, sizeof(mscclppSocketAddress));
//   setFilesLimit();

//   TRACE(MSCCLPP_INIT, "BEGIN");
//   /* Receive addresses from all ranks */
//   do {
//     mscclppSocket sock;
//     MSCCLPPCHECKGOTO(mscclppSocketInit(&sock), res, out);
//     MSCCLPPCHECKGOTO(mscclppSocketAccept(&sock, listenSock), res, out);
//     MSCCLPPCHECKGOTO(NetRecv(&sock, &info, sizeof(info)), res, out);
//     MSCCLPPCHECKGOTO(mscclppSocketClose(&sock), res, out);

//     if (nRanks != info.nRanks) {
//       WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nRanks, info.nRanks);
//       return;
//     }

//     if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(mscclppSocketAddress)) != 0) {
//       WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nRanks);
//       return;
//     }

//     // Save the connection handle for that rank
//     rankAddressesRoot[info.rank] = info.extAddressListenRoot;
//     rankAddresses[info.rank] = info.extAddressListen;

//     ++numCollected;
//     TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", info.rank, c, nranks);
//   } while (numCollected < nRanks);
//   TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", nranks);

//   // Send the connect handle for the next rank in the AllGather ring
//   for (int r = 0; r < nRanks; ++r) {
//     int next = (r + 1) % nRanks;
//     mscclppSocket sock;
//     MSCCLPPCHECKGOTO(mscclppSocketInit(&sock, &rankAddressesRoot[r], magic, mscclppSocketTypeBootstrap), res, out);
//     MSCCLPPCHECKGOTO(mscclppSocketConnect(&sock), res, out);
//     MSCCLPPCHECKGOTO(NetSend(&sock, &rankAddresses[next], sizeof(mscclppSocketAddress)), res, out);
//     MSCCLPPCHECKGOTO(mscclppSocketClose(&sock), res, out);
//   }
//   TRACE(MSCCLPP_INIT, "SENT OUT ALL %d HANDLES", nRanks);

// out:
//   if (listenSock != nullptr) {
//     mscclppSocketClose(listenSock);
//     free(listenSock);
//   }
//   TRACE(MSCCLPP_INIT, "DONE");
// }

// mscclppResult_t MscclppBootstrap::Impl::createRoot(mscclppBootstrap::UniqueId& handle)
// {
//   MSCCLPPCHECK(mscclppSocketInit(&this->listenSock, &handle.addr, handle.magic, mscclppSocketTypeBootstrap, NULL, 0));
//   MSCCLPPCHECK(mscclppSocketListen(&this->listenSock));
//   MSCCLPPCHECK(mscclppSocketGetAddr(&this->listenSock, &handle.addr));

//   std::thread thread(BootstrapRoot, listenSock, handle.magic, nRanks);
//   mscclppSetThreadName(thread.native_handle(), "MSCCLPP BootstrapR");
//   thread.detach();
//   return mscclppSuccess;
// }

// // Additional sync functions
// mscclppResult_t MscclppBootstrap::Impl::netSend(mscclppSocket* sock, void* data, int size)
// {
//   MSCCLPPCHECK(mscclppSocketSend(sock, &size, sizeof(int)));
//   MSCCLPPCHECK(mscclppSocketSend(sock, data, size));
//   return mscclppSuccess;
// }

// mscclppResult_t MscclppBootstrap::Impl::netRecv(mscclppSocket* sock, void* data, int size)
// {
//   int recvSize;
//   MSCCLPPCHECK(mscclppSocketRecv(sock, &recvSize, sizeof(int)));
//   if (recvSize > size) {
//     WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
//     return mscclppInternalError;
//   }
//   MSCCLPPCHECK(mscclppSocketRecv(sock, data, std::min(recvSize, size)));
//   return mscclppSuccess;
// }

// mscclppResult_t MscclppBootstrap::Impl::init(const mscclppComm& comm)
// {
//   this->rank = comm.rank;
//   this->nRanks = comm.nRanks;

//   mscclppSocket* proxySocket;
//   mscclppSocketAddress nextAddr;
//   mscclppSocket sock, listenSockRoot;
//   extInfo info;

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank, nranks);

//   info.rank = rank;
//   info.nRanks = this->nRanks;

//   uint64_t magic = this->handle.magic;
//   // Create socket for other ranks to contact me
//   MSCCLPPCHECK(
//     mscclppSocketInit(&this->listenSock, &bootstrapNetIfAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag));
//   MSCCLPPCHECK(mscclppSocketListen(&this->listenSock));
//   MSCCLPPCHECK(mscclppSocketGetAddr(&this->listenSock, &info.extAddressListen));

//   // Create socket for root to contact me
//   MSCCLPPCHECK(
//     mscclppSocketInit(&listenSockRoot, &bootstrapNetIfAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag));
//   MSCCLPPCHECK(mscclppSocketListen(&listenSockRoot));
//   MSCCLPPCHECK(mscclppSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

//   // stagger connection times to avoid an overload of the root
//   if (this->nRanks > 128) {
//     long msec = rank;
//     struct timespec tv;
//     tv.tv_sec = msec / 1000;
//     tv.tv_nsec = 1000000 * (msec % 1000);
//     TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
//     (void)nanosleep(&tv, NULL);
//   }

//   // send info on my listening socket to root
//   MSCCLPPCHECK(mscclppSocketInit(&sock, &this->handle.addr, magic, mscclppSocketTypeBootstrap, this->abortFlag));
//   MSCCLPPCHECK(mscclppSocketConnect(&sock));
//   MSCCLPPCHECK(NetSend(&sock, &info, sizeof(info)));
//   MSCCLPPCHECK(mscclppSocketClose(&sock));

//   // get info on my "next" rank in the bootstrap ring from root
//   MSCCLPPCHECK(mscclppSocketInit(&sock));
//   MSCCLPPCHECK(mscclppSocketAccept(&sock, &listenSockRoot));
//   MSCCLPPCHECK(NetRecv(&sock, &nextAddr, sizeof(union mscclppSocketAddress)));
//   MSCCLPPCHECK(mscclppSocketClose(&sock));
//   MSCCLPPCHECK(mscclppSocketClose(&listenSockRoot));

//   MSCCLPPCHECK(
//     mscclppSocketInit(&this->ringSendSocket, &nextAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag));
//   MSCCLPPCHECK(mscclppSocketConnect(&this->ringSendSocket));
//   // Accept the connect request from the previous rank in the AllGather ring
//   MSCCLPPCHECK(mscclppSocketInit(&this->ringRecvSocket));
//   MSCCLPPCHECK(mscclppSocketAccept(&this->ringRecvSocket, &this->listenSock));

//   // AllGather all listen handlers
//   MSCCLPPCHECK(mscclppCalloc(&this->peerCommAddresses, this->nRanks));
//   MSCCLPPCHECK(mscclppSocketGetAddr(&this->listenSock, this->peerCommAddresses + rank));
//   MSCCLPPCHECK(bootstrapAllGather(state, this->peerCommAddresses, sizeof(union mscclppSocketAddress)));

//   // Create the service proxy
//   MSCCLPPCHECK(mscclppCalloc(&this->peerProxyAddresses, this->nRanks));

//   // proxy is aborted through a message; don't set abortFlag
//   MSCCLPPCHECK(mscclppCalloc(&proxySocket, 1));
//   MSCCLPPCHECK(
//     mscclppSocketInit(proxySocket, &bootstrapNetIfAddr, comm->magic, mscclppSocketTypeProxy, comm->abortFlag));
//   MSCCLPPCHECK(mscclppSocketListen(proxySocket));
//   MSCCLPPCHECK(mscclppSocketGetAddr(proxySocket, &this->peerProxyAddresses[rank]));
//   MSCCLPPCHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union mscclppSocketAddress)));

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank, nranks);

//   return mscclppSuccess;
// }

// mscclppResult_t MscclppBootstrap::Impl::allGather(void* allData, int size)
// {
//   char* data = static_cast<char*>(allData);
//   int rank = this->rank;
//   int nRanks = this->nRanks;

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d", rank, nRanks, size);

//   /* Simple ring based AllGather
//    * At each step i receive data from (rank-i-1) from left
//    * and send previous step's data from (rank-i) to right
//    */
//   for (int i = 0; i < nRanks - 1; i++) {
//     size_t rSlice = (rank - i - 1 + nRanks) % nRanks;
//     size_t sSlice = (rank - i + nRanks) % nRanks;

//     // Send slice to the right
//     MSCCLPPCHECK(NetSend(&this->ringSendSocket, data + sSlice * size, size));
//     // Recv slice from the left
//     MSCCLPPCHECK(bootstrapNetRecv(&this->ringRecvSocket, data + rSlice * size, size));
//   }

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
//   return mscclppSuccess;
// }

//   mscclppResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size)
// {
//   mscclppResult_t ret = mscclppSuccess;
//   struct bootstrapState* state = (struct bootstrapState*)commState;
//   struct mscclppSocket sock;

//   MSCCLPPCHECKGOTO(mscclppSocketInit(&sock, state->peerCommAddresses + peer, state->magic, mscclppSocketTypeBootstrap,
//                                      state->abortFlag),
//                    ret, fail);
//   MSCCLPPCHECKGOTO(mscclppSocketConnect(&sock), ret, fail);
//   MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, &state->rank, sizeof(int)), ret, fail);
//   MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, &tag, sizeof(int)), ret, fail);
//   MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, data, size), ret, fail);

// exit:
//   MSCCLPPCHECK(mscclppSocketClose(&sock));
//   return ret;
// fail:
//   goto exit;
// }

// mscclppResult_t bootstrapBarrier(void* commState, int* ranks, int rank, int nranks, int tag)
// {
//   if (nranks == 1)
//     return mscclppSuccess;
//   TRACE(MSCCLPP_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

//   /* Simple intra process barrier
//    *
//    * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
//    * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
//    */
//   int data[1];
//   for (int mask = 1; mask < nranks; mask <<= 1) {
//     int src = (rank - mask + nranks) % nranks;
//     int dst = (rank + mask) % nranks;
//     MSCCLPPCHECK(bootstrapSend(commState, ranks[dst], tag, data, sizeof(data)));
//     MSCCLPPCHECK(bootstrapRecv(commState, ranks[src], tag, data, sizeof(data)));
//   }

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
//   return mscclppSuccess;
// }

// mscclppResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size)
// {
//   if (nranks == 1)
//     return mscclppSuccess;
//   char* data = (char*)allData;
//   TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

//   for (int i = 1; i < nranks; i++) {
//     int src = (rank - i + nranks) % nranks;
//     int dst = (rank + i) % nranks;
//     MSCCLPPCHECK(bootstrapSend(commState, ranks[dst], /*tag=*/i, data + rank * size, size));
//     MSCCLPPCHECK(bootstrapRecv(commState, ranks[src], /*tag=*/i, data + src * size, size));
//   }

//   TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
//   return mscclppSuccess;
// }

// mscclppResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct mscclppSocket* sock)
// {
//   // New unex
//   struct unexConn* unex;
//   MSCCLPPCHECK(mscclppCalloc(&unex, 1));
//   unex->peer = peer;
//   unex->tag = tag;
//   memcpy(&unex->sock, sock, sizeof(struct mscclppSocket));

//   // Enqueue
//   struct unexConn* list = state->unexpectedConnections;
//   if (list == NULL) {
//     state->unexpectedConnections = unex;
//     return mscclppSuccess;
//   }
//   while (list->next)
//     list = list->next;
//   list->next = unex;
//   return mscclppSuccess;
// }

// mscclppResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct mscclppSocket* sock,
//                                   int* found)
// {
//   struct unexConn* elem = state->unexpectedConnections;
//   struct unexConn* prev = NULL;
//   *found = 0;
//   while (elem) {
//     if (elem->peer == peer && elem->tag == tag) {
//       if (prev == NULL) {
//         state->unexpectedConnections = elem->next;
//       } else {
//         prev->next = elem->next;
//       }
//       memcpy(sock, &elem->sock, sizeof(struct mscclppSocket));
//       free(elem);
//       *found = 1;
//       return mscclppSuccess;
//     }
//     prev = elem;
//     elem = elem->next;
//   }
//   return mscclppSuccess;
// }

// static void unexpectedFree(struct bootstrapState* state)
// {
//   struct unexConn* elem = state->unexpectedConnections;
//   struct unexConn* prev = NULL;

//   while (elem) {
//     prev = elem;
//     elem = elem->next;
//     free(prev);
//   }
//   return;
// }

// // We can't know who we'll receive from, so we need to receive everything at once
// mscclppResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size)
// {
//   mscclppResult_t ret = mscclppSuccess;
//   struct bootstrapState* state = (struct bootstrapState*)commState;
//   struct mscclppSocket sock;
//   int newPeer, newTag;

//   // Search unexpected connections first
//   int found;
//   MSCCLPPCHECK(unexpectedDequeue(state, peer, tag, &sock, &found));
//   if (found) {
//     MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
//     goto exit;
//   }

//   // Then look for new connections
//   while (1) {
//     MSCCLPPCHECKGOTO(mscclppSocketInit(&sock), ret, fail);
//     MSCCLPPCHECKGOTO(mscclppSocketAccept(&sock, &state->listenSock), ret, fail);
//     MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, &newPeer, sizeof(int)), ret, fail);
//     MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, &newTag, sizeof(int)), ret, fail);
//     if (newPeer == peer && newTag == tag) {
//       MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
//       goto exit;
//     }
//     // Unexpected connection. Save for later.
//     MSCCLPPCHECKGOTO(unexpectedEnqueue(state, newPeer, newTag, &sock), ret, fail);
//   }
// exit:
//   MSCCLPPCHECK(mscclppSocketClose(&sock));
//   return ret;
// fail:
//   goto exit;
// }

// mscclppResult_t bootstrapClose(void* commState)
// {
//   struct bootstrapState* state = (struct bootstrapState*)commState;
//   if (state->unexpectedConnections != nullptr) {
//     unexpectedFree(state);
//     if (*state->abortFlag == 0) {
//       WARN("Unexpected connections are not empty");
//       return mscclppInternalError;
//     }
//   }

//   MSCCLPPCHECK(mscclppSocketClose(&state->listenSock));
//   MSCCLPPCHECK(mscclppSocketClose(&state->ringSendSocket));
//   MSCCLPPCHECK(mscclppSocketClose(&state->ringRecvSocket));

//   free(state->peerCommAddresses);
//   free(state);

//   return mscclppSuccess;
// }

// mscclppResult_t bootstrapAbort(void* commState)
// {
//   struct bootstrapState* state = (struct bootstrapState*)commState;
//   if (commState == nullptr)
//     return mscclppSuccess;
//   MSCCLPPCHECK(mscclppSocketClose(&state->listenSock));
//   MSCCLPPCHECK(mscclppSocketClose(&state->ringSendSocket));
//   MSCCLPPCHECK(mscclppSocketClose(&state->ringRecvSocket));
//   free(state->peerCommAddresses);
//   free(state->peerProxyAddresses);
//   free(state);
//   return mscclppSuccess;
// }

