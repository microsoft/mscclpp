#include <sys/resource.h>

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstring>
#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "api.h"
#include "config.hpp"
#include "debug.h"
#include "socket.h"
#include "utils_internal.hpp"

using namespace mscclpp;

static void setFilesLimit() {
  rlimit filesLimit;
  if (getrlimit(RLIMIT_NOFILE, &filesLimit) != 0) throw SysError("getrlimit failed", errno);
  filesLimit.rlim_cur = filesLimit.rlim_max;
  if (setrlimit(RLIMIT_NOFILE, &filesLimit) != 0) throw SysError("setrlimit failed", errno);
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

struct ExtInfo {
  int rank;
  int nRanks;
  mscclppSocketAddress extAddressListenRoot;
  mscclppSocketAddress extAddressListen;
};

MSCCLPP_API_CPP void BaseBootstrap::send(const std::vector<char>& data, int peer, int tag) {
  size_t size = data.size();
  send((void*)&size, sizeof(size_t), peer, tag);
  send((void*)data.data(), data.size(), peer, tag + 1);
}

MSCCLPP_API_CPP void BaseBootstrap::recv(std::vector<char>& data, int peer, int tag) {
  size_t size;
  recv((void*)&size, sizeof(size_t), peer, tag);
  data.resize(size);
  recv((void*)data.data(), data.size(), peer, tag + 1);
}

struct UniqueIdInternal {
  uint64_t magic;
  union mscclppSocketAddress addr;
};
static_assert(sizeof(UniqueIdInternal) <= sizeof(UniqueId), "UniqueIdInternal is too large to fit into UniqueId");

class Bootstrap::Impl {
 public:
  Impl(int rank, int nRanks);
  ~Impl();
  void initialize(const UniqueId uniqueId);
  void initialize(std::string ipPortPair);
  void establishConnections();
  UniqueId createUniqueId();
  UniqueId getUniqueId() const;
  int getRank();
  int getNranks();
  void allGather(void* allData, int size);
  void send(void* data, int size, int peer, int tag);
  void recv(void* data, int size, int peer, int tag);
  void barrier();
  void close();

 private:
  UniqueIdInternal uniqueId_;
  int rank_;
  int nRanks_;
  bool netInitialized;
  std::unique_ptr<Socket> listenSockRoot_;
  std::unique_ptr<Socket> listenSock_;
  std::unique_ptr<Socket> ringRecvSocket_;
  std::unique_ptr<Socket> ringSendSocket_;
  std::vector<mscclppSocketAddress> peerCommAddresses_;
  std::vector<int> barrierArr_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  volatile uint32_t* abortFlag_;
  std::thread rootThread_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
  mscclppSocketAddress netIfAddr_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash> peerSendSockets_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<Socket>, PairHash> peerRecvSockets_;

  void netSend(Socket* sock, const void* data, int size);
  void netRecv(Socket* sock, void* data, int size);

  std::shared_ptr<Socket> getPeerSendSocket(int peer, int tag);
  std::shared_ptr<Socket> getPeerRecvSocket(int peer, int tag);

  void bootstrapCreateRoot();
  void bootstrapRoot();
  void getRemoteAddresses(Socket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                          std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank);
  void sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                        const std::vector<mscclppSocketAddress>& rankAddressesRoot);
  void netInit(std::string ipPortPair);
};

Bootstrap::Impl::Impl(int rank, int nRanks)
    : rank_(rank),
      nRanks_(nRanks),
      netInitialized(false),
      peerCommAddresses_(nRanks, mscclppSocketAddress()),
      barrierArr_(nRanks, 0),
      abortFlagStorage_(new uint32_t(0)),
      abortFlag_(abortFlagStorage_.get()) {}

UniqueId Bootstrap::Impl::getUniqueId() const {
  UniqueId ret;
  std::memcpy(&ret, &uniqueId_, sizeof(uniqueId_));
  return ret;
}

UniqueId Bootstrap::Impl::createUniqueId() {
  netInit("");
  getRandomData(&uniqueId_.magic, sizeof(uniqueId_.magic));
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(mscclppSocketAddress));
  bootstrapCreateRoot();
  return getUniqueId();
}

int Bootstrap::Impl::getRank() { return rank_; }

int Bootstrap::Impl::getNranks() { return nRanks_; }

void Bootstrap::Impl::initialize(const UniqueId uniqueId) {
  netInit("");

  std::memcpy(&uniqueId_, &uniqueId, sizeof(uniqueId_));

  establishConnections();
}

void Bootstrap::Impl::initialize(std::string ipPortPair) {
  netInit(ipPortPair);

  uniqueId_.magic = 0xdeadbeef;
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(mscclppSocketAddress));
  mscclppSocketGetAddrFromString(&uniqueId_.addr, ipPortPair.c_str());

  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  establishConnections();
}

Bootstrap::Impl::~Impl() {
  if (abortFlag_) {
    *abortFlag_ = 1;
  }
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

void Bootstrap::Impl::getRemoteAddresses(Socket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                                         std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank) {
  ExtInfo info;
  mscclppSocketAddress zero;
  std::memset(&zero, 0, sizeof(mscclppSocketAddress));

  {
    Socket sock(nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, abortFlag_);
    sock.accept(listenSock);
    netRecv(&sock, &info, sizeof(info));
  }

  if (this->nRanks_ != info.nRanks) {
    throw Error("Bootstrap Root : mismatch in rank count from procs " + std::to_string(this->nRanks_) + " : " +
                    std::to_string(info.nRanks),
                ErrorCode::InternalError);
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(mscclppSocketAddress)) != 0) {
    throw Error("Bootstrap Root : rank " + std::to_string(info.rank) + " of " + std::to_string(this->nRanks_) +
                    " has already checked in",
                ErrorCode::InternalError);
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
}

void Bootstrap::Impl::sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                                       const std::vector<mscclppSocketAddress>& rankAddressesRoot) {
  int next = (peer + 1) % nRanks_;
  Socket sock(&rankAddressesRoot[peer], uniqueId_.magic, mscclppSocketTypeBootstrap, abortFlag_);
  sock.connect();
  netSend(&sock, &rankAddresses[next], sizeof(mscclppSocketAddress));
}

void Bootstrap::Impl::bootstrapCreateRoot() {
  listenSockRoot_ =
      std::make_unique<Socket>(&uniqueId_.addr, uniqueId_.magic, mscclppSocketTypeBootstrap, abortFlag_, 0);
  listenSockRoot_->listen();
  uniqueId_.addr = listenSockRoot_->getAddr();

  rootThread_ = std::thread([this]() {
    try {
      bootstrapRoot();
    } catch (const std::exception& e) {
      if (abortFlag_ && *abortFlag_) return;
      throw e;
    }
  });
}

void Bootstrap::Impl::bootstrapRoot() {
  int numCollected = 0;
  std::vector<mscclppSocketAddress> rankAddresses(nRanks_, mscclppSocketAddress());
  // for initial rank <-> root information exchange
  std::vector<mscclppSocketAddress> rankAddressesRoot(nRanks_, mscclppSocketAddress());

  std::memset(rankAddresses.data(), 0, sizeof(mscclppSocketAddress) * nRanks_);
  std::memset(rankAddressesRoot.data(), 0, sizeof(mscclppSocketAddress) * nRanks_);
  setFilesLimit();

  TRACE(MSCCLPP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int rank;
    getRemoteAddresses(listenSockRoot_.get(), rankAddresses, rankAddressesRoot, rank);
    ++numCollected;
    TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", rank, numCollected, nRanks_);
  } while (numCollected < nRanks_ && (!abortFlag_ || *abortFlag_ == 0));

  if (abortFlag_ && *abortFlag_) {
    TRACE(MSCCLPP_INIT, "ABORTED");
    return;
  }

  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < nRanks_; ++peer) {
    sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
  }

  TRACE(MSCCLPP_INIT, "DONE");
}

void Bootstrap::Impl::netInit(std::string ipPortPair) {
  if (netInitialized) return;
  if (!ipPortPair.empty()) {
    mscclppSocketAddress remoteAddr;
    mscclppSocketGetAddrFromString(&remoteAddr, ipPortPair.c_str());
    if (mscclppFindInterfaceMatchSubnet(netIfName_, &netIfAddr_, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      throw Error("NET/Socket : No usable listening interface found", ErrorCode::InternalError);
    }
  } else {
    int ret = mscclppFindInterfaces(netIfName_, &netIfAddr_, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      throw Error("Bootstrap : no socket interface found", ErrorCode::InternalError);
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName_);
  mscclppSocketToString(&netIfAddr_, line + strlen(line));
  INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
  netInitialized = true;
}

#define TIMEOUT(__exp)                                                   \
  do {                                                                   \
    try {                                                                \
      __exp;                                                             \
    } catch (const Error& e) {                                           \
      if (e.getErrorCode() == ErrorCode::Timeout) {                      \
        throw Error("Bootstrap connection timeout", ErrorCode::Timeout); \
      }                                                                  \
      throw;                                                             \
    }                                                                    \
  } while (0);

void Bootstrap::Impl::establishConnections() {
  const int64_t connectionTimeoutUs = (int64_t)Config::getInstance()->getBootstrapConnectionTimeoutConfig() * 1000000;
  Timer timer;
  mscclppSocketAddress nextAddr;
  ExtInfo info;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank_, nRanks_);

  auto getLeftTime = [&]() {
    int64_t timeout = connectionTimeoutUs - timer.elapsed();
    if (timeout <= 0) throw Error("Bootstrap connection timeout", ErrorCode::Timeout);
    return timeout;
  };

  info.rank = rank_;
  info.nRanks = nRanks_;

  uint64_t magic = uniqueId_.magic;
  // Create socket for other ranks to contact me
  listenSock_ = std::make_unique<Socket>(&netIfAddr_, magic, mscclppSocketTypeBootstrap, abortFlag_);
  listenSock_->listen();
  info.extAddressListen = listenSock_->getAddr();

  {
    // Create socket for root to contact me
    Socket lsock(&netIfAddr_, magic, mscclppSocketTypeBootstrap, abortFlag_);
    lsock.listen();
    info.extAddressListenRoot = lsock.getAddr();

    // stagger connection times to avoid an overload of the root
    auto randomSleep = [](int rank) {
      timespec tv;
      tv.tv_sec = rank / 1000;
      tv.tv_nsec = 1000000 * (rank % 1000);
      TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec", rank, rank);
      (void)nanosleep(&tv, NULL);
    };
    if (nRanks_ > 128) {
      randomSleep(rank_);
    }

    // send info on my listening socket to root
    {
      Socket sock(&uniqueId_.addr, magic, mscclppSocketTypeBootstrap, abortFlag_);
      TIMEOUT(sock.connect(getLeftTime()));
      netSend(&sock, &info, sizeof(info));
    }

    // get info on my "next" rank in the bootstrap ring from root
    {
      Socket sock(nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, abortFlag_);
      TIMEOUT(sock.accept(&lsock, getLeftTime()));
      netRecv(&sock, &nextAddr, sizeof(mscclppSocketAddress));
    }
  }

  ringSendSocket_ = std::make_unique<Socket>(&nextAddr, magic, mscclppSocketTypeBootstrap, abortFlag_);
  TIMEOUT(ringSendSocket_->connect(getLeftTime()));
  // Accept the connect request from the previous rank in the AllGather ring
  ringRecvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, abortFlag_);
  TIMEOUT(ringRecvSocket_->accept(listenSock_.get(), getLeftTime()));

  // AllGather all listen handlers
  peerCommAddresses_[rank_] = listenSock_->getAddr();
  allGather(peerCommAddresses_.data(), sizeof(mscclppSocketAddress));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank_, nRanks_);
}

void Bootstrap::Impl::allGather(void* allData, int size) {
  char* data = static_cast<char*>(allData);
  int rank = rank_;
  int nRanks = nRanks_;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d", rank, nRanks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nRanks - 1; i++) {
    size_t rSlice = (rank - i - 1 + nRanks) % nRanks;
    size_t sSlice = (rank - i + nRanks) % nRanks;

    // Send slice to the right
    netSend(ringSendSocket_.get(), data + sSlice * size, size);
    // Recv slice from the left
    netRecv(ringRecvSocket_.get(), data + rSlice * size, size);
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nRanks, size);
}

std::shared_ptr<Socket> Bootstrap::Impl::getPeerSendSocket(int peer, int tag) {
  auto it = peerSendSockets_.find(std::make_pair(peer, tag));
  if (it != peerSendSockets_.end()) {
    return it->second;
  }
  auto sock =
      std::make_shared<Socket>(&peerCommAddresses_[peer], uniqueId_.magic, mscclppSocketTypeBootstrap, abortFlag_);
  sock->connect();
  netSend(sock.get(), &rank_, sizeof(int));
  netSend(sock.get(), &tag, sizeof(int));
  peerSendSockets_[std::make_pair(peer, tag)] = sock;
  return sock;
}

std::shared_ptr<Socket> Bootstrap::Impl::getPeerRecvSocket(int peer, int tag) {
  auto it = peerRecvSockets_.find(std::make_pair(peer, tag));
  if (it != peerRecvSockets_.end()) {
    return it->second;
  }
  for (;;) {
    auto sock = std::make_shared<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, abortFlag_);
    sock->accept(listenSock_.get());
    int recvPeer, recvTag;
    netRecv(sock.get(), &recvPeer, sizeof(int));
    netRecv(sock.get(), &recvTag, sizeof(int));
    peerRecvSockets_[std::make_pair(recvPeer, recvTag)] = sock;
    if (recvPeer == peer && recvTag == tag) {
      return sock;
    }
  }
}

void Bootstrap::Impl::netSend(Socket* sock, const void* data, int size) {
  sock->send(&size, sizeof(int));
  sock->send(const_cast<void*>(data), size);
}

void Bootstrap::Impl::netRecv(Socket* sock, void* data, int size) {
  int recvSize;
  sock->recv(&recvSize, sizeof(int));
  if (recvSize > size) {
    std::stringstream ss;
    ss << "Message truncated : received " << recvSize << " bytes instead of " << size;
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }
  sock->recv(data, std::min(recvSize, size));
}

void Bootstrap::Impl::send(void* data, int size, int peer, int tag) {
  auto sock = getPeerSendSocket(peer, tag);
  netSend(sock.get(), data, size);
}

void Bootstrap::Impl::recv(void* data, int size, int peer, int tag) {
  auto sock = getPeerRecvSocket(peer, tag);
  netRecv(sock.get(), data, size);
}

void Bootstrap::Impl::barrier() { allGather(barrierArr_.data(), sizeof(int)); }

void Bootstrap::Impl::close() {
  listenSockRoot_.reset(nullptr);
  listenSock_.reset(nullptr);
  ringRecvSocket_.reset(nullptr);
  ringSendSocket_.reset(nullptr);
  peerSendSockets_.clear();
  peerRecvSockets_.clear();
}

MSCCLPP_API_CPP Bootstrap::Bootstrap(int rank, int nRanks) { pimpl_ = std::make_unique<Impl>(rank, nRanks); }

MSCCLPP_API_CPP UniqueId Bootstrap::createUniqueId() { return pimpl_->createUniqueId(); }

MSCCLPP_API_CPP UniqueId Bootstrap::getUniqueId() const { return pimpl_->getUniqueId(); }

MSCCLPP_API_CPP int Bootstrap::getRank() { return pimpl_->getRank(); }

MSCCLPP_API_CPP int Bootstrap::getNranks() { return pimpl_->getNranks(); }

MSCCLPP_API_CPP void Bootstrap::send(void* data, int size, int peer, int tag) { pimpl_->send(data, size, peer, tag); }

MSCCLPP_API_CPP void Bootstrap::recv(void* data, int size, int peer, int tag) { pimpl_->recv(data, size, peer, tag); }

MSCCLPP_API_CPP void Bootstrap::allGather(void* allData, int size) { pimpl_->allGather(allData, size); }

MSCCLPP_API_CPP void Bootstrap::initialize(UniqueId uniqueId) { pimpl_->initialize(uniqueId); }

MSCCLPP_API_CPP void Bootstrap::initialize(std::string ipPortPair) { pimpl_->initialize(ipPortPair); }

MSCCLPP_API_CPP void Bootstrap::barrier() { pimpl_->barrier(); }

MSCCLPP_API_CPP Bootstrap::~Bootstrap() { pimpl_->close(); }
