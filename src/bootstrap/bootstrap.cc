#include <sys/resource.h>
#include <sys/types.h>

#include <algorithm>
#include <cstring>
#include <list>
#include <mscclpp/core.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "api.h"
#include "checks_internal.hpp"
#include "socket.h"
#include "utils_internal.hpp"

using namespace mscclpp;

namespace {

mscclppResult_t setFilesLimit() {
  rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return mscclppSuccess;
}

}  // namespace

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

struct UnexpectedMsg {
  int peer;
  int tag;
  std::shared_ptr<mscclppSocket> sock;
};

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
  mscclppSocket listenSock_;
  mscclppSocket ringRecvSocket_;
  mscclppSocket ringSendSocket_;
  std::vector<mscclppSocketAddress> peerCommAddresses_;
  std::vector<int> barrierArr_;
  std::unique_ptr<uint32_t> abortFlagStorage_;
  volatile uint32_t* abortFlag_;
  std::thread rootThread_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
  mscclppSocketAddress netIfAddr_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<mscclppSocket>, PairHash> peerSendSockets_;
  std::unordered_map<std::pair<int, int>, std::shared_ptr<mscclppSocket>, PairHash> peerRecvSockets_;

  void netSend(mscclppSocket* sock, const void* data, int size);
  void netRecv(mscclppSocket* sock, void* data, int size);

  std::shared_ptr<mscclppSocket> getPeerSendSocket(int peer, int tag);
  std::shared_ptr<mscclppSocket> getPeerRecvSocket(int peer, int tag);

  void bootstrapCreateRoot();
  void bootstrapRoot(mscclppSocket listenSock);
  int getRemoteAddresses(mscclppSocket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
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
  MSCCLPPTHROW(mscclppSocketGetAddrFromString(&uniqueId_.addr, ipPortPair.c_str()));

  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  establishConnections();
}

Bootstrap::Impl::~Impl() {
  if (abortFlag_ != nullptr) {
    *abortFlag_ = 1;
  }
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

int Bootstrap::Impl::getRemoteAddresses(mscclppSocket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                                        std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank) {
  mscclppSocket sock;
  ExtInfo info;

  mscclppSocketAddress zero;
  std::memset(&zero, 0, sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketInit(&sock, nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, this->abortFlag_));
  if (mscclppSocketAccept(&sock, listenSock) == mscclppSuccess) {
    netRecv(&sock, &info, sizeof(info));
  }
  MSCCLPPTHROW(mscclppSocketClose(&sock));

  if (abortFlag_ && *abortFlag_) return 1;

  if (this->nRanks_ != info.nRanks) {
    throw mscclpp::Error("Bootstrap Root : mismatch in rank count from procs " + std::to_string(this->nRanks_) + " : " +
                             std::to_string(info.nRanks),
                         ErrorCode::InternalError);
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(mscclppSocketAddress)) != 0) {
    throw mscclpp::Error("Bootstrap Root : rank " + std::to_string(info.rank) + " of " + std::to_string(this->nRanks_) +
                             " has already checked in",
                         ErrorCode::InternalError);
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
  return 0;
}

void Bootstrap::Impl::sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                                       const std::vector<mscclppSocketAddress>& rankAddressesRoot) {
  mscclppSocket sock;
  int next = (peer + 1) % this->nRanks_;
  MSCCLPPTHROW(mscclppSocketInit(&sock, &rankAddressesRoot[peer], this->uniqueId_.magic, mscclppSocketTypeBootstrap,
                                 this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(&sock));
  netSend(&sock, &rankAddresses[next], sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketClose(&sock));
}

void Bootstrap::Impl::bootstrapCreateRoot() {
  mscclppSocket listenSock;

  MSCCLPPTHROW(mscclppSocketInit(&listenSock, &uniqueId_.addr, uniqueId_.magic, mscclppSocketTypeBootstrap,
                                 this->abortFlag_, 0));
  MSCCLPPTHROW(mscclppSocketListen(&listenSock));
  MSCCLPPTHROW(mscclppSocketGetAddr(&listenSock, &uniqueId_.addr));
  auto lambda = [this, listenSock]() { this->bootstrapRoot(listenSock); };
  rootThread_ = std::thread(lambda);
}

void Bootstrap::Impl::bootstrapRoot(mscclppSocket listenSock) {
  int numCollected = 0;
  std::vector<mscclppSocketAddress> rankAddresses(this->nRanks_, mscclppSocketAddress());
  // for initial rank <-> root information exchange
  std::vector<mscclppSocketAddress> rankAddressesRoot(this->nRanks_, mscclppSocketAddress());

  std::memset(rankAddresses.data(), 0, sizeof(mscclppSocketAddress) * this->nRanks_);
  std::memset(rankAddressesRoot.data(), 0, sizeof(mscclppSocketAddress) * this->nRanks_);
  setFilesLimit();

  TRACE(MSCCLPP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int rank;
    if (getRemoteAddresses(&listenSock, rankAddresses, rankAddressesRoot, rank) == 0) {
      ++numCollected;
      TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", rank, numCollected, this->nRanks_);
    }
  } while (numCollected < this->nRanks_ && (!abortFlag_ || *abortFlag_ == 0));

  if (abortFlag_ && *abortFlag_) {
    MSCCLPPTHROW(mscclppSocketClose(&listenSock));
    TRACE(MSCCLPP_INIT, "ABORTED");
    return;
  }

  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", this->nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < this->nRanks_; ++peer) {
    sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
  }

  MSCCLPPTHROW(mscclppSocketClose(&listenSock));
  TRACE(MSCCLPP_INIT, "DONE");
}

void Bootstrap::Impl::netInit(std::string ipPortPair) {
  if (netInitialized) return;
  if (!ipPortPair.empty()) {
    mscclppSocketAddress remoteAddr;
    if (mscclppSocketGetAddrFromString(&remoteAddr, ipPortPair.c_str()) != mscclppSuccess) {
      throw mscclpp::Error(
          "Invalid ipPortPair, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>",
          ErrorCode::InvalidUsage);
    }
    if (mscclppFindInterfaceMatchSubnet(netIfName_, &netIfAddr_, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      throw mscclpp::Error("NET/Socket : No usable listening interface found", ErrorCode::InternalError);
    }
  } else {
    int ret = mscclppFindInterfaces(netIfName_, &netIfAddr_, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      throw mscclpp::Error("Bootstrap : no socket interface found", ErrorCode::InternalError);
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName_);
  mscclppSocketToString(&netIfAddr_, line + strlen(line));
  INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
  netInitialized = true;
}

void Bootstrap::Impl::establishConnections() {
  mscclppSocketAddress nextAddr;
  mscclppSocket sock, listenSockRoot;
  ExtInfo info;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank_, nRanks_);

  info.rank = this->rank_;
  info.nRanks = this->nRanks_;

  uint64_t magic = this->uniqueId_.magic;
  // Create socket for other ranks to contact me
  MSCCLPPTHROW(mscclppSocketInit(&this->listenSock_, &netIfAddr_, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketListen(&this->listenSock_));
  MSCCLPPTHROW(mscclppSocketGetAddr(&this->listenSock_, &info.extAddressListen));

  // Create socket for root to contact me
  MSCCLPPTHROW(mscclppSocketInit(&listenSockRoot, &netIfAddr_, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketListen(&listenSockRoot));
  MSCCLPPTHROW(mscclppSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  auto randomSleep = [](int rank) {
    timespec tv;
    tv.tv_sec = rank / 1000;
    tv.tv_nsec = 1000000 * (rank % 1000);
    TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec", rank, rank);
    (void)nanosleep(&tv, NULL);
  };
  if (this->nRanks_ > 128) {
    randomSleep(this->rank_);
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName_);
  mscclppSocketToString(&this->uniqueId_.addr, line + strlen(line));

  // send info on my listening socket to root
  MSCCLPPTHROW(mscclppSocketInit(&sock, &this->uniqueId_.addr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(&sock));
  netSend(&sock, &info, sizeof(info));
  MSCCLPPTHROW(mscclppSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  MSCCLPPTHROW(mscclppSocketInit(&sock, nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketAccept(&sock, &listenSockRoot));
  netRecv(&sock, &nextAddr, sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketClose(&sock));
  MSCCLPPTHROW(mscclppSocketClose(&listenSockRoot));

  MSCCLPPTHROW(
      mscclppSocketInit(&this->ringSendSocket_, &nextAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(&this->ringSendSocket_));
  // Accept the connect request from the previous rank in the AllGather ring
  MSCCLPPTHROW(mscclppSocketInit(&this->ringRecvSocket_, nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown,
                                 this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketAccept(&this->ringRecvSocket_, &this->listenSock_));

  // AllGather all listen handlers
  MSCCLPPTHROW(mscclppSocketGetAddr(&this->listenSock_, &this->peerCommAddresses_[rank_]));
  allGather(this->peerCommAddresses_.data(), sizeof(mscclppSocketAddress));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank_, nRanks_);
}

void Bootstrap::Impl::allGather(void* allData, int size) {
  char* data = static_cast<char*>(allData);
  int rank = this->rank_;
  int nRanks = this->nRanks_;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d", rank, nRanks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nRanks - 1; i++) {
    size_t rSlice = (rank - i - 1 + nRanks) % nRanks;
    size_t sSlice = (rank - i + nRanks) % nRanks;

    // Send slice to the right
    netSend(&this->ringSendSocket_, data + sSlice * size, size);
    // Recv slice from the left
    netRecv(&this->ringRecvSocket_, data + rSlice * size, size);
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nRanks, size);
}

std::shared_ptr<mscclppSocket> Bootstrap::Impl::getPeerSendSocket(int peer, int tag) {
  auto it = this->peerSendSockets_.find(std::make_pair(peer, tag));
  if (it != this->peerSendSockets_.end()) {
    return it->second;
  }
  auto sock = std::make_shared<mscclppSocket>();
  MSCCLPPTHROW(mscclppSocketInit(sock.get(), &this->peerCommAddresses_[peer], this->uniqueId_.magic,
                                 mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(sock.get()));
  netSend(sock.get(), &this->rank_, sizeof(int));
  netSend(sock.get(), &tag, sizeof(int));
  this->peerSendSockets_[std::make_pair(peer, tag)] = sock;
  return sock;
}

std::shared_ptr<mscclppSocket> Bootstrap::Impl::getPeerRecvSocket(int peer, int tag) {
  auto it = this->peerRecvSockets_.find(std::make_pair(peer, tag));
  if (it != this->peerRecvSockets_.end()) {
    return it->second;
  }
  for (;;) {
    auto sock = std::make_shared<mscclppSocket>();
    MSCCLPPTHROW(
        mscclppSocketInit(sock.get(), nullptr, MSCCLPP_SOCKET_MAGIC, mscclppSocketTypeUnknown, this->abortFlag_));
    MSCCLPPTHROW(mscclppSocketAccept(sock.get(), &this->listenSock_));
    int recvPeer, recvTag;
    netRecv(sock.get(), &recvPeer, sizeof(int));
    netRecv(sock.get(), &recvTag, sizeof(int));
    this->peerRecvSockets_[std::make_pair(recvPeer, recvTag)] = sock;
    if (recvPeer == peer && recvTag == tag) {
      return sock;
    }
  }
}

void Bootstrap::Impl::netSend(mscclppSocket* sock, const void* data, int size) {
  MSCCLPPTHROW(mscclppSocketSend(sock, &size, sizeof(int)));
  MSCCLPPTHROW(mscclppSocketSend(sock, const_cast<void*>(data), size));
}

void Bootstrap::Impl::netRecv(mscclppSocket* sock, void* data, int size) {
  int recvSize;
  MSCCLPPTHROW(mscclppSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    throw mscclpp::Error(
        "Message truncated : received " + std::to_string(recvSize) + " bytes instead of " + std::to_string(size),
        ErrorCode::InvalidUsage);
  }
  MSCCLPPTHROW(mscclppSocketRecv(sock, data, std::min(recvSize, size)));
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
  MSCCLPPTHROW(mscclppSocketClose(&this->listenSock_));
  MSCCLPPTHROW(mscclppSocketClose(&this->ringSendSocket_));
  MSCCLPPTHROW(mscclppSocketClose(&this->ringRecvSocket_));
  for (auto& it : this->peerSendSockets_) {
    MSCCLPPTHROW(mscclppSocketClose(it.second.get()));
  }
  this->peerSendSockets_.clear();
  for (auto& it : this->peerRecvSockets_) {
    MSCCLPPTHROW(mscclppSocketClose(it.second.get()));
  }
  this->peerRecvSockets_.clear();
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
