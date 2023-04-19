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

struct bootstrapRootArgs
{
  struct mscclppSocket* listenSock;
  uint64_t magic;
};

struct mscclppBootstrap::UniqueId
{
  uint64_t magic;
  union mscclppSocketAddress addr;
};

struct unexConn
{
  int peer;
  int tag;
  struct mscclppSocket sock;
  struct unexConn* next;
};

struct bootstrapState
{
  struct mscclppSocket listenSock;
  struct mscclppSocket ringRecvSocket;
  struct mscclppSocket ringSendSocket;
  union mscclppSocketAddress* peerCommAddresses;
  union mscclppSocketAddress* peerProxyAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};

class mscclppBootstrap::Impl
{
public:
  static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
  static union mscclppSocketAddress bootstrapNetIfAddr;

  Impl() = default;

  static void* bootstrapRoot(void* args);
  mscclppResult_t NetInit(std::string ipPortPair = "");
  mscclppResult_t CreateRoot(mscclppBootstrap::UniqueId& handle);
};

mscclppResult_t mscclppBootstrap::Impl::NetInit(std::string ipPortPair)
{
  static std::atomic_bool initialized(false);
  static std::mutex initLock;

  if (initialized.load(std::memory_order_acquire)) {
    return mscclppSuccess;
  }
  std::lock_guard<std::mutex> lock(initLock);
  if (!initialized) {
    if (ipPortPair != "") {
      union mscclppSocketAddress remoteAddr;
      if (mscclppSocketGetAddrFromString(&remoteAddr, ipPortPair.c_str()) != mscclppSuccess) {
        WARN("Invalid MSCCLPP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
        return mscclppInvalidArgument;
      }
      if (mscclppFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <=
          0) {
        WARN("NET/Socket : No usable listening interface found");
        return mscclppSystemError;
      }
    } else {
      int nIfs = mscclppFindInterfaces(this->bootstrapNetIfName, &this->bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
      if (nIfs <= 0) {
        WARN("Bootstrap : no socket interface found");
        return mscclppInternalError;
      }
    }

    char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
    sprintf(line, " %s:", bootstrapNetIfName);
    mscclppSocketToString(&bootstrapNetIfAddr, line + strlen(line));
    INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
    initialized.store(true, std::memory_order_release);
  }
  return mscclppSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t
{
  findSubnetIf = -1,
  dontCareIf = -2
};

// Additional sync functions
static mscclppResult_t bootstrapNetSend(struct mscclppSocket* sock, void* data, int size)
{
  MSCCLPPCHECK(mscclppSocketSend(sock, &size, sizeof(int)));
  MSCCLPPCHECK(mscclppSocketSend(sock, data, size));
  return mscclppSuccess;
}
static mscclppResult_t bootstrapNetRecv(struct mscclppSocket* sock, void* data, int size)
{
  int recvSize;
  MSCCLPPCHECK(mscclppSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return mscclppInternalError;
  }
  MSCCLPPCHECK(mscclppSocketRecv(sock, data, std::min(recvSize, size)));
  return mscclppSuccess;
}

struct extInfo
{
  int rank;
  int nranks;
  union mscclppSocketAddress extAddressListenRoot;
  union mscclppSocketAddress extAddressListen;
};

void* mscclppBootstrap::Impl::bootstrapRoot(void* bootstrapArgs)
{
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)bootstrapArgs;
  struct mscclppSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  mscclppResult_t res = mscclppSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union mscclppSocketAddress* rankAddresses = NULL;
  union mscclppSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  union mscclppSocketAddress* zero = NULL;
  MSCCLPPCHECKGOTO(mscclppCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(MSCCLPP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    struct mscclppSocket sock;
    MSCCLPPCHECKGOTO(mscclppSocketInit(&sock), res, out);
    MSCCLPPCHECKGOTO(mscclppSocketAccept(&sock, listenSock), res, out);
    MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, out);
    MSCCLPPCHECKGOTO(mscclppSocketClose(&sock), res, out);

    if (c == 0) {
      nranks = info.nranks;
      MSCCLPPCHECKGOTO(mscclppCalloc(&rankAddresses, nranks), res, out);
      MSCCLPPCHECKGOTO(mscclppCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union mscclppSocketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot + info.rank, &info.extAddressListenRoot, sizeof(union mscclppSocketAddress));
    memcpy(rankAddresses + info.rank, &info.extAddressListen, sizeof(union mscclppSocketAddress));

    ++c;
    TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", info.rank, c, nranks);
  } while (c < nranks);
  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r = 0; r < nranks; ++r) {
    int next = (r + 1) % nranks;
    struct mscclppSocket sock;
    MSCCLPPCHECKGOTO(mscclppSocketInit(&sock, rankAddressesRoot + r, magic, mscclppSocketTypeBootstrap), res, out);
    MSCCLPPCHECKGOTO(mscclppSocketConnect(&sock), res, out);
    MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, rankAddresses + next, sizeof(union mscclppSocketAddress)), res, out);
    MSCCLPPCHECKGOTO(mscclppSocketClose(&sock), res, out);
  }
  TRACE(MSCCLPP_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  if (listenSock != NULL) {
    mscclppSocketClose(listenSock);
    free(listenSock);
  }
  if (rankAddresses)
    free(rankAddresses);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  if (zero)
    free(zero);
  free(bootstrapArgs);

  TRACE(MSCCLPP_INIT, "DONE");
  return NULL;
}

mscclppResult_t mscclppBootstrap::Impl::CreateRoot(mscclppBootstrap::UniqueId& handle)
{
  struct mscclppSocket* listenSock;
  struct bootstrapRootArgs* args;

  MSCCLPPCHECK(mscclppCalloc(&listenSock, 1));
  MSCCLPPCHECK(mscclppSocketInit(listenSock, &handle.addr, handle.magic, mscclppSocketTypeBootstrap, NULL, 0));
  MSCCLPPCHECK(mscclppSocketListen(listenSock));
  MSCCLPPCHECK(mscclppSocketGetAddr(listenSock, &handle.addr));

  MSCCLPPCHECK(mscclppCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle.magic;

  std::thread thread(bootstrapRoot, (void*)args);
  mscclppSetThreadName(thread.native_handle(), "MSCCLPP BootstrapR");
  thread.detach();
  return mscclppSuccess;
}

std::unique_ptr<mscclppBootstrap::UniqueId> mscclppBootstrap::GetUniqueId()
{
  pimpl->NetInit();

  mscclppBootstrap::UniqueId handle;
  mscclppResult_t ret = getRandomData(&handle.magic, sizeof(handle.magic));
  if (ret != mscclppSuccess) {
    throw std::runtime_error("getting random data failed");
  }
  std::memcpy(&handle.addr, &pimpl->bootstrapNetIfAddr, sizeof(union mscclppSocketAddress));
  ret = pimpl->CreateRoot(handle);
  if (ret != mscclppSuccess) {
    throw std::runtime_error("Bootstrap : failed to get unique ID");
  }
  return std::make_unique<mscclppBootstrap::UniqueId>(handle);
}

mscclppResult_t bootstrapInit(struct mscclppBootstrapHandle* handle, struct mscclppComm* comm)
{
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct bootstrapState* state;
  struct mscclppSocket* proxySocket;
  mscclppSocketAddress nextAddr;
  struct mscclppSocket sock, listenSockRoot;
  struct extInfo info;

  MSCCLPPCHECK(mscclppCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = comm->abortFlag;
  comm->bootstrap = state;
  comm->magic = state->magic = handle->magic;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = rank;
  info.nranks = nranks;

  // Create socket for other ranks to contact me
  MSCCLPPCHECK(mscclppSocketInit(&state->listenSock, &bootstrapNetIfAddr, comm->magic, mscclppSocketTypeBootstrap,
                                 comm->abortFlag));
  MSCCLPPCHECK(mscclppSocketListen(&state->listenSock));
  MSCCLPPCHECK(mscclppSocketGetAddr(&state->listenSock, &info.extAddressListen));

  // Create socket for root to contact me
  MSCCLPPCHECK(
    mscclppSocketInit(&listenSockRoot, &bootstrapNetIfAddr, comm->magic, mscclppSocketTypeBootstrap, comm->abortFlag));
  MSCCLPPCHECK(mscclppSocketListen(&listenSockRoot));
  MSCCLPPCHECK(mscclppSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void)nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  MSCCLPPCHECK(mscclppSocketInit(&sock, &handle->addr, comm->magic, mscclppSocketTypeBootstrap, comm->abortFlag));
  MSCCLPPCHECK(mscclppSocketConnect(&sock));
  MSCCLPPCHECK(bootstrapNetSend(&sock, &info, sizeof(info)));
  MSCCLPPCHECK(mscclppSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  MSCCLPPCHECK(mscclppSocketInit(&sock));
  MSCCLPPCHECK(mscclppSocketAccept(&sock, &listenSockRoot));
  MSCCLPPCHECK(bootstrapNetRecv(&sock, &nextAddr, sizeof(union mscclppSocketAddress)));
  MSCCLPPCHECK(mscclppSocketClose(&sock));
  MSCCLPPCHECK(mscclppSocketClose(&listenSockRoot));

  MSCCLPPCHECK(
    mscclppSocketInit(&state->ringSendSocket, &nextAddr, comm->magic, mscclppSocketTypeBootstrap, comm->abortFlag));
  MSCCLPPCHECK(mscclppSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  MSCCLPPCHECK(mscclppSocketInit(&state->ringRecvSocket));
  MSCCLPPCHECK(mscclppSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  MSCCLPPCHECK(mscclppCalloc(&state->peerCommAddresses, nranks));
  MSCCLPPCHECK(mscclppSocketGetAddr(&state->listenSock, state->peerCommAddresses + rank));
  MSCCLPPCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union mscclppSocketAddress)));

  // Create the service proxy
  MSCCLPPCHECK(mscclppCalloc(&state->peerProxyAddresses, nranks));

  // proxy is aborted through a message; don't set abortFlag
  MSCCLPPCHECK(mscclppCalloc(&proxySocket, 1));
  MSCCLPPCHECK(
    mscclppSocketInit(proxySocket, &bootstrapNetIfAddr, comm->magic, mscclppSocketTypeProxy, comm->abortFlag));
  MSCCLPPCHECK(mscclppSocketListen(proxySocket));
  MSCCLPPCHECK(mscclppSocketGetAddr(proxySocket, state->peerProxyAddresses + rank));
  MSCCLPPCHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union mscclppSocketAddress)));
  // MSCCLPPCHECK(mscclppProxyInit(comm, proxySocket, state->peerProxyAddresses));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return mscclppSuccess;
}

mscclppResult_t bootstrapAllGather(void* commState, void* allData, int size)
{
  struct bootstrapState* state = (struct bootstrapState*)commState;
  char* data = (char*)allData;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    MSCCLPPCHECK(bootstrapNetSend(&state->ringSendSocket, data + sslice * size, size));
    // Recv slice from the left
    MSCCLPPCHECK(bootstrapNetRecv(&state->ringRecvSocket, data + rslice * size, size));
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return mscclppSuccess;
}

mscclppResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size)
{
  mscclppResult_t ret = mscclppSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct mscclppSocket sock;

  MSCCLPPCHECKGOTO(mscclppSocketInit(&sock, state->peerCommAddresses + peer, state->magic, mscclppSocketTypeBootstrap,
                                     state->abortFlag),
                   ret, fail);
  MSCCLPPCHECKGOTO(mscclppSocketConnect(&sock), ret, fail);
  MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, &state->rank, sizeof(int)), ret, fail);
  MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, &tag, sizeof(int)), ret, fail);
  MSCCLPPCHECKGOTO(bootstrapNetSend(&sock, data, size), ret, fail);

exit:
  MSCCLPPCHECK(mscclppSocketClose(&sock));
  return ret;
fail:
  goto exit;
}

mscclppResult_t bootstrapBarrier(void* commState, int* ranks, int rank, int nranks, int tag)
{
  if (nranks == 1)
    return mscclppSuccess;
  TRACE(MSCCLPP_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  /* Simple intra process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1];
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    MSCCLPPCHECK(bootstrapSend(commState, ranks[dst], tag, data, sizeof(data)));
    MSCCLPPCHECK(bootstrapRecv(commState, ranks[src], tag, data, sizeof(data)));
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return mscclppSuccess;
}

mscclppResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size)
{
  if (nranks == 1)
    return mscclppSuccess;
  char* data = (char*)allData;
  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  for (int i = 1; i < nranks; i++) {
    int src = (rank - i + nranks) % nranks;
    int dst = (rank + i) % nranks;
    MSCCLPPCHECK(bootstrapSend(commState, ranks[dst], /*tag=*/i, data + rank * size, size));
    MSCCLPPCHECK(bootstrapRecv(commState, ranks[src], /*tag=*/i, data + src * size, size));
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return mscclppSuccess;
}

mscclppResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct mscclppSocket* sock)
{
  // New unex
  struct unexConn* unex;
  MSCCLPPCHECK(mscclppCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct mscclppSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return mscclppSuccess;
  }
  while (list->next)
    list = list->next;
  list->next = unex;
  return mscclppSuccess;
}

mscclppResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct mscclppSocket* sock,
                                  int* found)
{
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct mscclppSocket));
      free(elem);
      *found = 1;
      return mscclppSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return mscclppSuccess;
}

static void unexpectedFree(struct bootstrapState* state)
{
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
mscclppResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size)
{
  mscclppResult_t ret = mscclppSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct mscclppSocket sock;
  int newPeer, newTag;

  // Search unexpected connections first
  int found;
  MSCCLPPCHECK(unexpectedDequeue(state, peer, tag, &sock, &found));
  if (found) {
    MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
    goto exit;
  }

  // Then look for new connections
  while (1) {
    MSCCLPPCHECKGOTO(mscclppSocketInit(&sock), ret, fail);
    MSCCLPPCHECKGOTO(mscclppSocketAccept(&sock, &state->listenSock), ret, fail);
    MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, &newPeer, sizeof(int)), ret, fail);
    MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, &newTag, sizeof(int)), ret, fail);
    if (newPeer == peer && newTag == tag) {
      MSCCLPPCHECKGOTO(bootstrapNetRecv(&sock, ((char*)data), size), ret, fail);
      goto exit;
    }
    // Unexpected connection. Save for later.
    MSCCLPPCHECKGOTO(unexpectedEnqueue(state, newPeer, newTag, &sock), ret, fail);
  }
exit:
  MSCCLPPCHECK(mscclppSocketClose(&sock));
  return ret;
fail:
  goto exit;
}

mscclppResult_t bootstrapClose(void* commState)
{
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (*state->abortFlag == 0) {
      WARN("Unexpected connections are not empty");
      return mscclppInternalError;
    }
  }

  MSCCLPPCHECK(mscclppSocketClose(&state->listenSock));
  MSCCLPPCHECK(mscclppSocketClose(&state->ringSendSocket));
  MSCCLPPCHECK(mscclppSocketClose(&state->ringRecvSocket));

  free(state->peerCommAddresses);
  free(state);

  return mscclppSuccess;
}

mscclppResult_t bootstrapAbort(void* commState)
{
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (commState == NULL)
    return mscclppSuccess;
  MSCCLPPCHECK(mscclppSocketClose(&state->listenSock));
  MSCCLPPCHECK(mscclppSocketClose(&state->ringSendSocket));
  MSCCLPPCHECK(mscclppSocketClose(&state->ringRecvSocket));
  free(state->peerCommAddresses);
  free(state->peerProxyAddresses);
  free(state);
  return mscclppSuccess;
}
