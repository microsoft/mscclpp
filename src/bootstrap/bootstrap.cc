#include "bootstrap.h"
#include "api.h"
#include "checks.hpp"
#include "mscclpp.hpp"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/types.h>

using namespace mscclpp;

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
  rlimit filesLimit;
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

struct UnexpectedMsg
{
  int peer;
  int tag;
  std::shared_ptr<mscclppSocket> sock;
};

struct ExtInfo
{
  int rank;
  int nRanks;
  mscclppSocketAddress extAddressListenRoot;
  mscclppSocketAddress extAddressListen;
};

struct UniqueIdInternal
{
  uint64_t magic;
  union mscclppSocketAddress addr;
};
static_assert(sizeof(UniqueIdInternal) <= sizeof(UniqueId), "UniqueIdInternal is too large to fit into UniqueId");

class Bootstrap::Impl
{
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
  std::list<UnexpectedMsg> unexpectedMessages_;
  std::vector<int> barrierArr_;
  volatile uint32_t* abortFlag_;
  std::thread rootThread_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
  mscclppSocketAddress netIfAddr_;

  void netSend(mscclppSocket* sock, const void* data, int size);
  void netRecv(mscclppSocket* sock, void* data, int size);

  void bootstrapCreateRoot();
  void bootstrapRoot(mscclppSocket listenSock);
  void getRemoteAddresses(mscclppSocket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                          std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank);
  void sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                        const std::vector<mscclppSocketAddress>& rankAddressesRoot);
  void netInit(std::string ipPortPair);
};

// UniqueId MscclppBootstrap::Impl::uniqueId_;

Bootstrap::Impl::Impl(int rank, int nRanks)
  : rank_(rank), nRanks_(nRanks), netInitialized(false), peerCommAddresses_(nRanks, mscclppSocketAddress()),
    barrierArr_(nRanks, 0), abortFlag_(nullptr)
{
}

UniqueId Bootstrap::Impl::getUniqueId() const
{
  UniqueId ret;
  std::memcpy(&ret, &uniqueId_, sizeof(uniqueId_));
  return ret;
}

UniqueId Bootstrap::Impl::createUniqueId()
{
  netInit("");
  MSCCLPPTHROW(getRandomData(&uniqueId_.magic, sizeof(uniqueId_.magic)));
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(mscclppSocketAddress));
  bootstrapCreateRoot();
  return getUniqueId();
}

int Bootstrap::Impl::getRank()
{
  return rank_;
}

int Bootstrap::Impl::getNranks()
{
  return nRanks_;
}

void Bootstrap::Impl::initialize(const UniqueId uniqueId)
{
  netInit("");

  std::memcpy(&uniqueId_, &uniqueId, sizeof(uniqueId_));

  establishConnections();
}

void Bootstrap::Impl::initialize(std::string ipPortPair)
{
  netInit(ipPortPair);

  uniqueId_.magic = 0xdeadbeef;
  std::memcpy(&uniqueId_.addr, &netIfAddr_, sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketGetAddrFromString(&uniqueId_.addr, ipPortPair.c_str()));

  if (rank_ == 0) {
    bootstrapCreateRoot();
  }

  establishConnections();
}

Bootstrap::Impl::~Impl()
{
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

void Bootstrap::Impl::getRemoteAddresses(mscclppSocket* listenSock, std::vector<mscclppSocketAddress>& rankAddresses,
                                         std::vector<mscclppSocketAddress>& rankAddressesRoot, int& rank)
{
  mscclppSocket sock;
  ExtInfo info;

  mscclppSocketAddress zero;
  std::memset(&zero, 0, sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketInit(&sock));
  MSCCLPPTHROW(mscclppSocketAccept(&sock, listenSock));
  netRecv(&sock, &info, sizeof(info));
  MSCCLPPTHROW(mscclppSocketClose(&sock));

  if (this->nRanks_ != info.nRanks) {
    throw std::runtime_error("Bootstrap Root : mismatch in rank count from procs " + std::to_string(this->nRanks_) +
                             " : " + std::to_string(info.nRanks));
  }

  if (std::memcmp(&zero, &rankAddressesRoot[info.rank], sizeof(mscclppSocketAddress)) != 0) {
    throw std::runtime_error("Bootstrap Root : rank " + std::to_string(info.rank) + " of " +
                             std::to_string(this->nRanks_) + " has already checked in");
  }

  // Save the connection handle for that rank
  rankAddressesRoot[info.rank] = info.extAddressListenRoot;
  rankAddresses[info.rank] = info.extAddressListen;
  rank = info.rank;
}

void Bootstrap::Impl::sendHandleToPeer(int peer, const std::vector<mscclppSocketAddress>& rankAddresses,
                                       const std::vector<mscclppSocketAddress>& rankAddressesRoot)
{
  mscclppSocket sock;
  int next = (peer + 1) % this->nRanks_;
  MSCCLPPTHROW(mscclppSocketInit(&sock, &rankAddressesRoot[peer], this->uniqueId_.magic, mscclppSocketTypeBootstrap));
  MSCCLPPTHROW(mscclppSocketConnect(&sock));
  netSend(&sock, &rankAddresses[next], sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketClose(&sock));
}

void Bootstrap::Impl::bootstrapCreateRoot()
{
  mscclppSocket listenSock;

  // mscclppSocket* listenSock = new mscclppSocket(); // TODO(saemal) make this a shared ptr
  MSCCLPPTHROW(
    mscclppSocketInit(&listenSock, &uniqueId_.addr, uniqueId_.magic, mscclppSocketTypeBootstrap, nullptr, 0));
  MSCCLPPTHROW(mscclppSocketListen(&listenSock));
  MSCCLPPTHROW(mscclppSocketGetAddr(&listenSock, &uniqueId_.addr));
  auto lambda = [this, listenSock]() { this->bootstrapRoot(listenSock); };
  rootThread_ = std::thread(lambda);
}

void Bootstrap::Impl::bootstrapRoot(mscclppSocket listenSock)
{
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
    getRemoteAddresses(&listenSock, rankAddresses, rankAddressesRoot, rank);
    ++numCollected;
    TRACE(MSCCLPP_INIT, "Received connect from rank %d total %d/%d", rank, numCollected, this->nRanks_);
  } while (numCollected < this->nRanks_);
  TRACE(MSCCLPP_INIT, "COLLECTED ALL %d HANDLES", this->nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int peer = 0; peer < this->nRanks_; ++peer) {
    sendHandleToPeer(peer, rankAddresses, rankAddressesRoot);
  }

  TRACE(MSCCLPP_INIT, "DONE");
}

void Bootstrap::Impl::netInit(std::string ipPortPair)
{
  if (netInitialized)
    return;
  if (!ipPortPair.empty()) {
    mscclppSocketAddress remoteAddr;
    if (mscclppSocketGetAddrFromString(&remoteAddr, ipPortPair.c_str()) != mscclppSuccess) {
      throw std::runtime_error(
        "Invalid ipPortPair, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
    }
    if (mscclppFindInterfaceMatchSubnet(netIfName_, &netIfAddr_, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      throw std::runtime_error("NET/Socket : No usable listening interface found");
    }
  } else {
    int ret = mscclppFindInterfaces(netIfName_, &netIfAddr_, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) {
      throw std::runtime_error("Bootstrap : no socket interface found");
    }
  }

  char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
  std::sprintf(line, " %s:", netIfName_);
  mscclppSocketToString(&netIfAddr_, line + strlen(line));
  INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
  netInitialized = true;
}

void Bootstrap::Impl::establishConnections()
{
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
  MSCCLPPTHROW(mscclppSocketInit(&sock));
  MSCCLPPTHROW(mscclppSocketAccept(&sock, &listenSockRoot));
  netRecv(&sock, &nextAddr, sizeof(mscclppSocketAddress));
  MSCCLPPTHROW(mscclppSocketClose(&sock));
  MSCCLPPTHROW(mscclppSocketClose(&listenSockRoot));

  MSCCLPPTHROW(
    mscclppSocketInit(&this->ringSendSocket_, &nextAddr, magic, mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(&this->ringSendSocket_));
  // Accept the connect request from the previous rank in the AllGather ring
  MSCCLPPTHROW(mscclppSocketInit(&this->ringRecvSocket_));
  MSCCLPPTHROW(mscclppSocketAccept(&this->ringRecvSocket_, &this->listenSock_));

  // AllGather all listen handlers
  MSCCLPPTHROW(mscclppSocketGetAddr(&this->listenSock_, &this->peerCommAddresses_[rank_]));
  allGather(this->peerCommAddresses_.data(), sizeof(mscclppSocketAddress));

  TRACE(MSCCLPP_INIT, "rank %d nranks %d - DONE", rank_, nRanks_);
}

void Bootstrap::Impl::allGather(void* allData, int size)
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
    netSend(&this->ringSendSocket_, data + sSlice * size, size);
    // Recv slice from the left
    netRecv(&this->ringRecvSocket_, data + rSlice * size, size);
  }

  TRACE(MSCCLPP_INIT, "rank %d nranks %d size %d - DONE", rank, nRanks, size);
}

void Bootstrap::Impl::netSend(mscclppSocket* sock, const void* data, int size)
{
  MSCCLPPTHROW(mscclppSocketSend(sock, &size, sizeof(int)));
  MSCCLPPTHROW(mscclppSocketSend(sock, const_cast<void*>(data), size));
}

void Bootstrap::Impl::netRecv(mscclppSocket* sock, void* data, int size)
{
  int recvSize;
  MSCCLPPTHROW(mscclppSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    throw std::runtime_error("Message truncated : received " + std::to_string(recvSize) + " bytes instead of " +
                             std::to_string(size));
  }
  MSCCLPPTHROW(mscclppSocketRecv(sock, data, std::min(recvSize, size)));
}

void Bootstrap::Impl::send(void* data, int size, int peer, int tag)
{
  mscclppSocket sock;
  MSCCLPPTHROW(mscclppSocketInit(&sock, &this->peerCommAddresses_[peer], this->uniqueId_.magic,
                                 mscclppSocketTypeBootstrap, this->abortFlag_));
  MSCCLPPTHROW(mscclppSocketConnect(&sock));
  netSend(&sock, &this->rank_, sizeof(int));
  netSend(&sock, &tag, sizeof(int));
  netSend(&sock, data, size);

  MSCCLPPTHROW(mscclppSocketClose(&sock));
}

void Bootstrap::Impl::recv(void* data, int size, int peer, int tag)
{
  // search over all unexpected messages
  auto lambda = [peer, tag](const UnexpectedMsg& msg) { return msg.peer == peer && msg.tag == tag; };
  auto it = std::find_if(unexpectedMessages_.begin(), unexpectedMessages_.end(), lambda);
  if (it != unexpectedMessages_.end()) {
    // found a match
    netRecv(it->sock.get(), data, size);
    MSCCLPPTHROW(mscclppSocketClose(it->sock.get()));
    unexpectedMessages_.erase(it);
    return;
  }
  // didn't find one
  while (true) {
    auto sock = std::make_shared<mscclppSocket>();
    int newPeer, newTag;
    MSCCLPPTHROW(mscclppSocketInit(sock.get()));
    MSCCLPPTHROW(mscclppSocketAccept(sock.get(), &this->listenSock_));
    netRecv(sock.get(), &newPeer, sizeof(int));
    netRecv(sock.get(), &newTag, sizeof(int));
    if (newPeer == peer && newTag == tag) {
      netRecv(sock.get(), ((char*)data), size);
      MSCCLPPTHROW(mscclppSocketClose(sock.get()));
      return;
    }
    // Unexpected message. Save for later.
    unexpectedMessages_.push_back({newPeer, newTag, sock});
  }
}

void Bootstrap::Impl::barrier()
{
  allGather(barrierArr_.data(), sizeof(int));
}

void Bootstrap::Impl::close()
{
  MSCCLPPTHROW(mscclppSocketClose(&this->listenSock_));
  MSCCLPPTHROW(mscclppSocketClose(&this->ringSendSocket_));
  MSCCLPPTHROW(mscclppSocketClose(&this->ringRecvSocket_));
}

MSCCLPP_API_CPP Bootstrap::Bootstrap(int rank, int nRanks)
{
  // pimpl_ = std::make_unique<Impl>(ipPortPair, rank, nRanks, uniqueId);
  pimpl_ = std::make_unique<Impl>(rank, nRanks);
}

MSCCLPP_API_CPP UniqueId Bootstrap::createUniqueId()
{
  return pimpl_->createUniqueId();
}

MSCCLPP_API_CPP UniqueId Bootstrap::getUniqueId() const
{
  return pimpl_->getUniqueId();
}

MSCCLPP_API_CPP int Bootstrap::getRank()
{
  return pimpl_->getRank();
}

MSCCLPP_API_CPP int Bootstrap::getNranks()
{
  return pimpl_->getNranks();
}

MSCCLPP_API_CPP void Bootstrap::send(void* data, int size, int peer, int tag)
{
  pimpl_->send(data, size, peer, tag);
}

MSCCLPP_API_CPP void Bootstrap::recv(void* data, int size, int peer, int tag)
{
  pimpl_->recv(data, size, peer, tag);
}

MSCCLPP_API_CPP void Bootstrap::allGather(void* allData, int size)
{
  pimpl_->allGather(allData, size);
}

MSCCLPP_API_CPP void Bootstrap::initialize(UniqueId uniqueId)
{
  pimpl_->initialize(uniqueId);
}

MSCCLPP_API_CPP void Bootstrap::initialize(std::string ipPortPair)
{
  pimpl_->initialize(ipPortPair);
}

MSCCLPP_API_CPP void Bootstrap::barrier()
{
  pimpl_->barrier();
}

MSCCLPP_API_CPP Bootstrap::~Bootstrap()
{
  pimpl_->close();
}

// ------------------- Old bootstrap functions -------------------
struct BootstrapRootArgs
{
  struct mscclppSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
static union mscclppSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

mscclppResult_t bootstrapNetInit(const char* ip_port_pair)
{
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      const char* env;
      if (ip_port_pair) {
        env = ip_port_pair;
      } else {
        env = getenv("MSCCLPP_COMM_ID");
      }
      if (env) {
        union mscclppSocketAddress remoteAddr;
        if (mscclppSocketGetAddrFromString(&remoteAddr, env) != mscclppSuccess) {
          WARN("Invalid MSCCLPP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return mscclppInvalidArgument;
        }
        if (mscclppFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                            1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return mscclppSystemError;
        }
      } else {
        int nIfs = mscclppFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return mscclppInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      sprintf(line, " %s:", bootstrapNetIfName);
      mscclppSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(MSCCLPP_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return mscclppSuccess;
}

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

// struct ExtInfo
// {
//   int rank;
//   int nranks;
//   union mscclppSocketAddress extAddressListenRoot;
//   union mscclppSocketAddress extAddressListen;
// };

#include <sys/resource.h>

// static mscclppResult_t setFilesLimit()
// {
//   struct rlimit filesLimit;
//   SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
//   filesLimit.rlim_cur = filesLimit.rlim_max;
//   SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
//   return mscclppSuccess;
// }

static void* bootstrapRoot(void* rargs)
{
  struct BootstrapRootArgs* args = (struct BootstrapRootArgs*)rargs;
  struct mscclppSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  mscclppResult_t res = mscclppSuccess;
  int nranks = 0, c = 0;
  struct ExtInfo info;
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
      nranks = info.nRanks;
      MSCCLPPCHECKGOTO(mscclppCalloc(&rankAddresses, nranks), res, out);
      MSCCLPPCHECKGOTO(mscclppCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nRanks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nRanks);
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
  free(rargs);

  TRACE(MSCCLPP_INIT, "DONE");
  return NULL;
}

mscclppResult_t bootstrapCreateRoot(struct mscclppBootstrapHandle* handle)
{
  struct mscclppSocket* listenSock;
  struct BootstrapRootArgs* args;
  pthread_t thread;

  MSCCLPPCHECK(mscclppCalloc(&listenSock, 1));
  MSCCLPPCHECK(mscclppSocketInit(listenSock, &handle->addr, handle->magic, mscclppSocketTypeBootstrap, NULL, 0));
  MSCCLPPCHECK(mscclppSocketListen(listenSock));
  MSCCLPPCHECK(mscclppSocketGetAddr(listenSock, &handle->addr));

  MSCCLPPCHECK(mscclppCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  NEQCHECK(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), 0);
  mscclppSetThreadName(thread, "MSCCLPP BootstrapR");
  NEQCHECK(pthread_detach(thread), 0); // will not be pthread_join()'d
  return mscclppSuccess;
}

// #include <netinet/in.h>
// #include <arpa/inet.h>

mscclppResult_t bootstrapGetUniqueId(struct mscclppBootstrapHandle* handle, bool isRoot, const char* ip_port_pair)
{
  memset(handle, 0, sizeof(mscclppBootstrapHandle));
  const char* env = NULL;

  if (ip_port_pair) {
    env = ip_port_pair;
  } else {
    env = getenv("MSCCLPP_COMM_ID");
  }
  if (env) {
    handle->magic = 0xdeadbeef;

    INFO(MSCCLPP_ENV, "MSCCLPP_COMM_ID set by environment to %s", env);
    if (mscclppSocketGetAddrFromString(&handle->addr, env) != mscclppSuccess) {
      WARN("Invalid MSCCLPP_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return mscclppInvalidArgument;
    }
    if (isRoot)
      MSCCLPPCHECK(bootstrapCreateRoot(handle));
  } else {
    MSCCLPPCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union mscclppSocketAddress));
    MSCCLPPCHECK(bootstrapCreateRoot(handle));
  }
  printf("addr = %s port = %d\n", inet_ntoa(handle->addr.sin.sin_addr), (int)ntohs(handle->addr.sin.sin_port));
  // printf("addr = %s\n", inet_ntoa((*(struct sockaddr_in*)&handle->addr.sa).sin_addr));

  return mscclppSuccess;
}

struct UnexConn
{
  int peer;
  int tag;
  struct mscclppSocket sock;
  struct UnexConn* next;
};

struct BootstrapState
{
  struct mscclppSocket listenSock;
  struct mscclppSocket ringRecvSocket;
  struct mscclppSocket ringSendSocket;
  union mscclppSocketAddress* peerCommAddresses;
  union mscclppSocketAddress* peerProxyAddresses;
  struct UnexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};

mscclppResult_t bootstrapInit(struct mscclppBootstrapHandle* handle, struct mscclppComm* comm)
{
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct BootstrapState* state;
  struct mscclppSocket* proxySocket;
  mscclppSocketAddress nextAddr;
  struct mscclppSocket sock, listenSockRoot;
  struct ExtInfo info;

  MSCCLPPCHECK(mscclppCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = comm->abortFlag;
  comm->bootstrap = state;
  comm->magic = state->magic = handle->magic;

  TRACE(MSCCLPP_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = rank;
  info.nRanks = nranks;

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
  struct BootstrapState* state = (struct BootstrapState*)commState;
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
  struct BootstrapState* state = (struct BootstrapState*)commState;
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

mscclppResult_t unexpectedEnqueue(struct BootstrapState* state, int peer, int tag, struct mscclppSocket* sock)
{
  // New unex
  struct UnexConn* unex;
  MSCCLPPCHECK(mscclppCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct mscclppSocket));

  // Enqueue
  struct UnexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return mscclppSuccess;
  }
  while (list->next)
    list = list->next;
  list->next = unex;
  return mscclppSuccess;
}

mscclppResult_t unexpectedDequeue(struct BootstrapState* state, int peer, int tag, struct mscclppSocket* sock,
                                  int* found)
{
  struct UnexConn* elem = state->unexpectedConnections;
  struct UnexConn* prev = NULL;
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

static void unexpectedFree(struct BootstrapState* state)
{
  struct UnexConn* elem = state->unexpectedConnections;
  struct UnexConn* prev = NULL;

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
  struct BootstrapState* state = (struct BootstrapState*)commState;
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
  struct BootstrapState* state = (struct BootstrapState*)commState;
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
  struct BootstrapState* state = (struct BootstrapState*)commState;
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