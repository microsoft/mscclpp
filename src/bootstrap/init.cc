#include "mscclpp.h"
#include "bootstrap.h"
#include "core.h"
#include <map>
#include <sstream>

static uint64_t hashUniqueId(mscclppUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(mscclppUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
// static size_t maxLocalSizeBytes = 0;

static mscclppResult_t mscclppInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) return mscclppSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    // initEnv();
    // initGdrCopy();
    // maxLocalSizeBytes = mscclppKernMaxLocalSize();
    // int carveout = mscclppParamL1SharedMemoryCarveout();
    // if (carveout) mscclppKernSetSharedMemoryCarveout(carveout);
    // Always initialize bootstrap network
    MSCCLPPCHECK(bootstrapNetInit());
    // MSCCLPPCHECK(mscclppNetPluginInit());

    // initNvtxRegisteredEnums();
    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&initLock);
  return mscclppSuccess;
}

static std::string mscclppShmFileName(mscclppComm_t comm, int rank)
{
  std::stringstream ss;
  ss << "mscclpp." << std::hex << comm->magic << "." << rank;
  return ss.str();
}

mscclppResult_t mscclppGetUniqueId(mscclppUniqueId* out) {
  MSCCLPPCHECK(mscclppInit());
//   mscclppCHECK(PtrCheck(out, "GetUniqueId", "out"));
  mscclppResult_t res = bootstrapGetUniqueId((struct mscclppBootstrapHandle*)out);
  TRACE_CALL("mscclppGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}

MSCCLPP_API(mscclppResult_t, mscclppBootStrapAllGather, mscclppComm_t comm, void* data, int size);
mscclppResult_t mscclppBootStrapAllGather(mscclppComm_t comm, void* data, int size){
  MSCCLPPCHECK(bootstrapAllGather(comm->bootstrap, data, size));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppCommInitRank, mscclppComm_t* comm, int nranks, int rank, const char* ip_port_pair);
mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, int rank, const char* ip_port_pair){
  mscclppResult_t res = mscclppSuccess;
  mscclppComm_t _comm = NULL;
  uint64_t hash = getHostHash();
  uint64_t *hashes;
  std::map<uint64_t, int> hashToNode;

  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm, 1), res, fail);
  _comm->rank = rank;
  _comm->nRanks = nranks;

  MSCCLPPCHECK(bootstrapNetInit(ip_port_pair));
  mscclppBootstrapHandle handle;
  MSCCLPPCHECK(bootstrapGetUniqueId(&handle, rank == 0, ip_port_pair));
  _comm->magic = handle.magic;

  MSCCLPPCHECKGOTO(mscclppCudaHostCalloc((uint32_t **)&_comm->abortFlag, 1), res, fail);
  MSCCLPPCHECK(bootstrapInit(&handle, _comm));

  _comm->maxLocalRanks = 8;
  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm->rankToNode, nranks), res, fail);
  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm->rankToLocalRank, nranks), res, fail);
  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm->localRankToRank, _comm->maxLocalRanks), res, fail);

  MSCCLPPCHECKGOTO(mscclppCalloc(&hashes, nranks), res, fail);
  hashes[rank] = hash;
  MSCCLPPCHECK(bootstrapAllGather(_comm->bootstrap, hashes, sizeof(uint64_t)));

  for (int i = 0; i < nranks; ++i) {
    auto it = hashToNode.find(hashes[i]);
    if (it == hashToNode.end()) {
      _comm->nNodes++;
      hashToNode[hashes[i]] = _comm->nNodes - 1;
      _comm->rankToNode[i] = _comm->nNodes - 1;
    } else {
      _comm->rankToNode[i] = it->second;
    }
    if (hashes[i] == hash) {
      _comm->rankToLocalRank[i] = _comm->localRanks++;
      _comm->localRankToRank[_comm->rankToLocalRank[i]] = i;
    }
  }
  if (_comm->localRanks > _comm->maxLocalRanks) {
    WARN("Too many ranks on the same host: %d", _comm->localRanks);
    res = mscclppInvalidUsage;
    goto fail;
  }
  _comm->node = _comm->rankToNode[rank];
  _comm->localRank = _comm->rankToLocalRank[rank];

  *comm = _comm;
  return res;
fail:
  if (_comm) {
    if (_comm->abortFlag) mscclppCudaHostFree((void *)_comm->abortFlag);
    free(_comm);
  }
  if (comm) *comm = NULL;
  return res;
}

MSCCLPP_API(mscclppResult_t, mscclppCommDestroy, mscclppComm_t comm);
mscclppResult_t mscclppCommDestroy(mscclppComm_t comm){
  if (comm == NULL)
    return mscclppSuccess;

  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i]) {
      MSCCLPPCHECK(mscclppIbContextDestroy(comm->ibContext[i]));
    }
  }

  if (comm->bootstrap)
    MSCCLPPCHECK(bootstrapClose(comm->bootstrap));

  mscclppCudaHostFree((void *)comm->abortFlag);
  free(comm);
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppConnect, mscclppComm_t comm, int remoteRank,
            void *buff, size_t buffSize, int *flag, int tag, mscclppTransport_t transportType, const char *ibDev);
mscclppResult_t mscclppConnect(mscclppComm_t comm, int remoteRank, void *buff, size_t buffSize,
                               int *flag, int tag, mscclppTransport_t transportType, const char *ibDev/*=NULL*/)
{
  struct mscclppConn *conn = &comm->conns[comm->nConns++];
  conn->transport = transportType;
  conn->remoteRank = remoteRank;
  conn->tag = tag;
  conn->buff = buff;
  conn->buffSize = buffSize;
  conn->flag = flag;
  conn->ibCtx = NULL;
  conn->ibQp = NULL;

  if (ibDev != NULL) {
    // Check if an IB context exists
    int ibDevIdx = -1;
    int firstNullIdx = -1;
    for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
      if (comm->ibContext[i] == NULL) {
        if (firstNullIdx == -1) {
          firstNullIdx = i;
        }
      } else if (strncmp(comm->ibContext[i]->ctx->device->name, ibDev, IBV_SYSFS_NAME_MAX) == 0) {
        ibDevIdx = i;
        break;
      }
    }
    if (ibDevIdx == -1) {
      // Create a new context.
      if (firstNullIdx == -1) {
        WARN("Too many IB devices");
        return mscclppInvalidUsage;
      }
      ibDevIdx = firstNullIdx;
      if (mscclppIbContextCreate(&comm->ibContext[ibDevIdx], ibDev) != mscclppSuccess) {
        WARN("Failed to create IB context");
        return mscclppInternalError;
      }
    }
    conn->ibCtx = comm->ibContext[ibDevIdx];
  }
  return mscclppSuccess;
}

struct connInfo {
  cudaIpcMemHandle_t handleBuff;
  cudaIpcMemHandle_t handleFlag;
  mscclppIbQpInfo infoQp;
  mscclppIbMrInfo infoBuffMr;
  mscclppIbMrInfo infoLocalFlagMr;
  mscclppIbMrInfo infoRemoteFlagMr;
};

MSCCLPP_API(mscclppResult_t, mscclppConnectionSetup, mscclppComm_t comm);
mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm)
{
  // Allocate connection info to be shared with GPU
  MSCCLPPCHECK(mscclppCudaHostCalloc(&comm->devConns, comm->nConns));

  // Send info to peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    struct mscclppDevConn *devConn = &comm->devConns[i];
    conn->devConn = devConn;
    devConn->tag = conn->tag;
    devConn->localBuff = conn->buff;
    devConn->localFlag = conn->flag;
    MSCCLPPCHECK(mscclppCudaHostCalloc(&devConn->trigger, 1));

    struct connInfo cInfo;
    if (conn->transport == mscclppTransportP2P) {
      CUDACHECK(cudaIpcGetMemHandle(&cInfo.handleBuff, devConn->localBuff));
      CUDACHECK(cudaIpcGetMemHandle(&cInfo.handleFlag, devConn->localFlag));
    } else if (conn->transport == mscclppTransportIB) {
      devConn->remoteBuff = NULL;
      MSCCLPPCHECK(mscclppCudaCalloc(&devConn->remoteFlag, 1));

      struct mscclppIbContext *ibCtx = conn->ibCtx;
      if (conn->ibQp == NULL) {
        MSCCLPPCHECK(mscclppIbContextCreateQp(ibCtx, &conn->ibQp));
      }
      MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->localBuff, conn->buffSize, &conn->ibBuffMr));
      MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->localFlag, sizeof(int), &conn->ibLocalFlagMr));
      MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->remoteFlag, sizeof(int), &conn->ibRemoteFlagMr));
      cInfo.infoQp = conn->ibQp->info;
      cInfo.infoBuffMr = conn->ibBuffMr->info;
      cInfo.infoLocalFlagMr = conn->ibLocalFlagMr->info;
      cInfo.infoRemoteFlagMr = conn->ibRemoteFlagMr->info;
    }
    MSCCLPPCHECK(bootstrapSend(comm->bootstrap, conn->remoteRank, conn->tag, &cInfo, sizeof(cInfo)));
  }

  // Recv info from peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    struct mscclppDevConn *devConn = &comm->devConns[i];

    struct connInfo cInfo;
    MSCCLPPCHECK(bootstrapRecv(comm->bootstrap, conn->remoteRank, conn->tag, &cInfo, sizeof(cInfo)));
    if (conn->transport == mscclppTransportP2P) {
      CUDACHECK(cudaIpcOpenMemHandle(&devConn->remoteBuff, cInfo.handleBuff, cudaIpcMemLazyEnablePeerAccess));
      CUDACHECK(cudaIpcOpenMemHandle((void **)&devConn->remoteFlag, cInfo.handleFlag, cudaIpcMemLazyEnablePeerAccess));
    } else if (conn->transport == mscclppTransportIB) {
      if (conn->ibQp->rtr(&cInfo.infoQp) != 0) {
        WARN("Failed to transition QP to RTR");
        return mscclppInvalidUsage;
      }
      if (conn->ibQp->rts() != 0) {
        WARN("Failed to transition QP to RTS");
        return mscclppInvalidUsage;
      }
      conn->ibBuffMrInfo = cInfo.infoBuffMr;
      conn->ibLocalFlagMrInfo = cInfo.infoLocalFlagMr;
      conn->ibRemoteFlagMrInfo = cInfo.infoRemoteFlagMr;
    }
  }

  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppProxyLaunch, mscclppComm_t comm);
mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm)
{
  MSCCLPPCHECK(mscclppProxyCreate(comm));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppProxyStop, mscclppComm_t comm);
mscclppResult_t mscclppProxyStop(mscclppComm_t comm)
{
  MSCCLPPCHECK(mscclppProxyDestroy(comm));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppGetLocalRank, mscclppComm_t comm, int *localRank);
mscclppResult_t mscclppGetLocalRank(mscclppComm_t comm, int *localRank)
{
  *localRank = comm->localRank;
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppGetNodeFromRank, mscclppComm_t comm, int rank, int *node);
mscclppResult_t mscclppGetNodeFromRank(mscclppComm_t comm, int rank, int *node)
{
  *node = comm->rankToNode[rank];
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppGetDevConns, mscclppComm_t comm, mscclppDevConn_t* devConns);
mscclppResult_t mscclppGetDevConns(mscclppComm_t comm, mscclppDevConn_t* devConns)
{
  *devConns = comm->devConns;
  return mscclppSuccess;
}
