#include "bootstrap.h"
#include "utils.h"

#include <cstring>
#include <mutex>
#include <queue>
#include <thread>

#include <sys/resource.h>
#include <sys/types.h>

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

  Impl(std::string ipPortPair, int rank, int nRanks, const mscclppBootstrapHandle handle);
  ~Impl();
  mscclppResult_t initialize();
  mscclppResult_t allGather(void* allData, int size);
  mscclppResult_t send(void* data, int size, int peer, int tag);
  mscclppResult_t recv(void* data, int size, int peer, int tag);
  mscclppResult_t barrier();

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
  std::thread rootThread_;

  static mscclppResult_t netSend(mscclppSocket* sock, const void* data, int size);
  static mscclppResult_t netRecv(mscclppSocket* sock, void* data, int size);

  mscclppResult_t bootstrapRoot();
  mscclppResult_t getRemoteAddresses(mscclppSocket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                                     std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank);
  mscclppResult_t sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                                   const std::vector<mscclppSocketAddress>& rankAddressesRoot);
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

  if (!ipPortPair.empty()) {
    uniqueId_.magic = 0xdeadbeef;
  } else {
    mscclppResult_t ret = getRandomData(&uniqueId_.magic, sizeof(uniqueId_.magic));
    if (ret != mscclppSuccess) {
      throw std::runtime_error("getting random data failed");
    }
  }
  std::memcpy(&uniqueId_.addr, &bootstrapNetIfAddr, sizeof(union mscclppSocketAddress));
  if (rank_ == 0) {
    rootThread_ = std::thread(&MscclppBootstrap::Impl::bootstrapRoot, this, &listenSock_, uniqueId_.magic, nRanks_);
  }
}

MscclppBootstrap::Impl::~Impl()
{
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

mscclppResult_t MscclppBootstrap::Impl::getRemoteAddresses(mscclppSocket* listenSock,
                                                           std::vector<mscclppSocketAddress>& rankAddresses,
                                                           std::vector<mscclppSocketAddress>& rankAddressesRoot,
                                                           int& rank)
{
  mscclppSocket sock;
  extInfo info;
  mscclppResult_t res = mscclppSuccess;

  mscclppSocketAddress zero;
  std::memset(&zero, 0, sizeof(mscclppSocketAddress));
  res = mscclppSocketInit(&sock);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketInit failed");
    return res;
  }
  res = mscclppSocketAccept(&sock, listenSock);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketAccept failed");
    return res;
  }
  res = netRecv(&sock, &info, sizeof(info));
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : netRecv failed");
    return res;
  }
  res = mscclppSocketClose(&sock);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketClose failed");
    return res;
  }

  if (this->nRanks_ != info.nRanks) {
    WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", this->nRanks_, info.nRanks);
    return res;
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(mscclppSocketAddress)) != 0) {
    WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, this->nRanks_);
    return res;
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
  return res;
}

mscclppResult_t MscclppBootstrap::Impl::sendHandleToPeer(int peer,
                                                         const std::vector<mscclppSocketAddress>& rankAddresses,
                                                         const std::vector<mscclppSocketAddress>& rankAddressesRoot)
{
  mscclppSocket sock;
  mscclppResult_t res;
  int next = (peer + 1) % this->nRanks_;
  res = mscclppSocketInit(&sock, &rankAddressesRoot[peer], this->uniqueId_.magic, mscclppSocketTypeBootstrap);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketInit failed");
    return res;
  }
  res = mscclppSocketConnect(&sock);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketConnect failed");
    return res;
  }
  res = netSend(&sock, &rankAddresses[next], sizeof(mscclppSocketAddress));
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : netSend failed");
    return res;
  }
  res = mscclppSocketClose(&sock);
  if (res != mscclppSuccess) {
    WARN("Bootstrap Root : mscclppSocketClose failed");
    return res;
  }
  return mscclppSuccess;
}

mscclppResult_t MscclppBootstrap::Impl::bootstrapRoot()
{
  mscclppResult_t res = mscclppSuccess;
  int numCollected = 0;
  std::vector<mscclppSocketAddress> rankAddresses(this->nRanks_, mscclppSocketAddress());
  // for initial rank <-> root information exchange
  std::vector<mscclppSocketAddress> rankAddressesRoot(this->nRanks_, mscclppSocketAddress());

  std::memset(rankAddresses.data(), 0, sizeof(mscclppSocketAddress) * this->nRanks_);
  std::memset(rankAddressesRoot.data(), 0, sizeof(mscclppSocketAddress) * this->nRanks_);
  setFilesLimit();

  mscclppSocket listenSock;
  MSCCLPPCHECK(
    mscclppSocketInit(&listenSock, &uniqueId_.addr, uniqueId_.magic, mscclppSocketTypeBootstrap, nullptr, 0));
  MSCCLPPCHECK(mscclppSocketListen(&listenSock));

  TRACE(MSCCLPP_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int rank;
    res = getRemoteAddresses(&listenSock, rankAddresses, rankAddressesRoot, rank);
    if (res != mscclppSuccess) {
      WARN("Bootstrap Root : getRemoteAddresses failed");
      break;
    }
    ++numCollected;
    TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", rank, numCollected, this->nRanks_);
  } while (numCollected < this->nRanks_);
  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", this->nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < this->nRanks_; ++peer) {
    res = sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
    if (res != mscclppSuccess) {
      WARN("Bootstrap Root : sendHandleToPeer failed");
      break;
    }
  }
  if (res == mscclppSuccess) {
    TRACE(MSCCLPP_INIT, "SENT OUT ALL %d HANDLES", this->nRanks_);
  }
  TRACE(MSCCLPP_INIT, "DONE");
  return res;
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

mscclppResult_t MscclppBootstrap::Impl::initialize()
{
  mscclppSocket* proxySocket;
  mscclppSocketAddress nextAddr;
  mscclppSocket sock, listenSockRoot;
  extInfo info;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = this->rank_;
  info.nRanks = this->nRanks_;

  uint64_t magic = this->uniqueId_.magic;
  // Create socket for other ranks to contact me
  MSCCLPPCHECK(
    mscclppSocketInit(&this->listenSock_, &bootstrapNetIfAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketListen(&this->listenSock_));
  MSCCLPPCHECK(mscclppSocketGetAddr(&this->listenSock_, &info.extAddressListen));

  // Create socket for root to contact me
  MSCCLPPCHECK(
    mscclppSocketInit(&listenSockRoot, &bootstrapNetIfAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketListen(&listenSockRoot));
  MSCCLPPCHECK(mscclppSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  auto randomSleep = [](int rank) {
    struct timespec tv;
    tv.tv_sec = rank / 1000;
    tv.tv_nsec = 1000000 * (rank % 1000);
    TRACE(MSCCLPP_INIT, "rank %d delaying connection to root by %ld msec", rank, rank);
    (void)nanosleep(&tv, NULL);
  };
  if (this->nRanks_ > 128) {
    randomSleep(this->rank_);
  }

  // send info on my listening socket to root
  MSCCLPPCHECK(mscclppSocketInit(&sock, &this->uniqueId_.addr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketConnect(&sock));
  MSCCLPPCHECK(netSend(&sock, &info, sizeof(info)));
  MSCCLPPCHECK(mscclppSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  MSCCLPPCHECK(mscclppSocketInit(&sock));
  MSCCLPPCHECK(mscclppSocketAccept(&sock, &listenSockRoot));
  MSCCLPPCHECK(netRecv(&sock, &nextAddr, sizeof(union mscclppSocketAddress)));
  MSCCLPPCHECK(mscclppSocketClose(&sock));
  MSCCLPPCHECK(mscclppSocketClose(&listenSockRoot));

  MSCCLPPCHECK(
    mscclppSocketInit(&this->ringSendSocket_, &nextAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketConnect(&this->ringSendSocket_));
  // Accept the connect request from the previous rank in the AllGather ring
  MSCCLPPCHECK(mscclppSocketInit(&this->ringRecvSocket_));
  MSCCLPPCHECK(mscclppSocketAccept(&this->ringRecvSocket_, &this->listenSock_));

  // AllGather all listen handlers
  MSCCLPPCHECK(mscclppSocketGetAddr(&this->listenSock_, &this->peerCommAddresses_[rank_]));
  MSCCLPPCHECK(allGather(this->peerCommAddresses_.data(), sizeof(union mscclppSocketAddress)));

  // proxy is aborted through a message; don't set abortFlag
  MSCCLPPCHECK(mscclppCalloc(&proxySocket, 1));
  MSCCLPPCHECK(mscclppSocketInit(proxySocket, &bootstrapNetIfAddr, magic, mscclppSocketTypeProxy, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketListen(proxySocket));
  MSCCLPPCHECK(mscclppSocketGetAddr(proxySocket, &this->peerProxyAddresses_[rank_]));
  MSCCLPPCHECK(allGather(this->peerProxyAddresses_.data(), sizeof(union mscclppSocketAddress)));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return mscclppSuccess;
}

mscclppResult_t MscclppBootstrap::Impl::allGather(void* allData, int size)
{
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
    MSCCLPPCHECK(netSend(&this->ringSendSocket_, data + sSlice * size, size));
    // Recv slice from the left
    MSCCLPPCHECK(netRecv(&this->ringRecvSocket_, data + rSlice * size, size));
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return mscclppSuccess;
}

mscclppResult_t MscclppBootstrap::Impl::netSend(mscclppSocket* sock, const void* data, int size)
{
  MSCCLPPCHECK(mscclppSocketSend(sock, &size, sizeof(int)));
  MSCCLPPCHECK(mscclppSocketSend(sock, const_cast<void*>(data), size));
  return mscclppSuccess;
}

mscclppResult_t MscclppBootstrap::Impl::netRecv(mscclppSocket* sock, void* data, int size)
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

mscclppResult_t MscclppBootstrap::Impl::send(void* data, int size, int peer, int tag)
{
  mscclppSocket sock;
  MSCCLPPCHECK(mscclppSocketInit(&sock, &this->peerCommAddresses_[peer], this->uniqueId_.magic,
                                 mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPCHECK(mscclppSocketConnect(&sock));
  MSCCLPPCHECK(netSend(&sock, &this->rank_, sizeof(int)));
  MSCCLPPCHECK(netSend(&sock, &tag, sizeof(int)));
  MSCCLPPCHECK(netSend(&sock, data, size));

  MSCCLPPCHECK(mscclppSocketClose(&sock));
}

MscclppBootstrap::MscclppBootstrap(std::string ipPortPair, int rank, int nRanks)
{
  pimpl_ = std::make_unique<Impl>(ipPortPair, rank, nRanks, mscclppBootstrapHandle{0});
}

MscclppBootstrap::MscclppBootstrap(mscclppBootstrapHandle handle, int rank, int nRanks)
{
  pimpl_ = std::make_unique<Impl>("", rank, nRanks, handle);
}

MscclppBootstrap::UniqueId MscclppBootstrap::GetUniqueId()
{
  return pimpl_->uniqueId_;
}

void MscclppBootstrap::Send(void* data, int size, int peer, int tag)
{
  mscclppResult_t res = pimpl_->send(data, size, peer, tag);
  if (res != mscclppSuccess) {
    throw std::runtime_error("MscclppBootstrap::Send failed");
  }
}

void MscclppBootstrap::Recv(void* data, int size, int peer, int tag)
{
  mscclppResult_t res = pimpl_->recv(data, size, peer, tag);
  if (res != mscclppSuccess) {
    throw std::runtime_error("MscclppBootstrap::Recv failed");
  }
}

void MscclppBootstrap::AllGather(void* allData, int size)
{
  mscclppResult_t res = pimpl_->allGather(allData, size);
  if (res != mscclppSuccess) {
    throw std::runtime_error("MscclppBootstrap::AllGather failed");
  }
}

void MscclppBootstrap::Initialize()
{
  mscclppResult_t res = pimpl_->initialize();
  if (res != mscclppSuccess) {
    throw std::runtime_error("MscclppBootstrap::Initialize failed");
  }
}

void MscclppBootstrap::Barrier()
{
  mscclppResult_t res = pimpl_->barrier();
  if (res != mscclppSuccess) {
    throw std::runtime_error("MscclppBootstrap::Barrier failed");
  }
}
