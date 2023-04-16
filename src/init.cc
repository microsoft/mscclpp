#include "bootstrap.h"
#include "config.h"
#include "core.h"
#if defined(MSCCLPP_USE_GDRCOPY)
#include "gdr.h"
#endif
#include "mscclpp.h"
#include <map>
#include <sstream>
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

static uint64_t hashUniqueId(mscclppUniqueId const& id)
{
  char const* bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for (int i = 0; i < (int)sizeof(mscclppUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
// static size_t maxLocalSizeBytes = 0;

#if defined(MSCCLPP_USE_GDRCOPY)

gdr_t mscclppGdrCopy = NULL;

mscclppResult_t initGdrCopy()
{
  if (mscclppGdrCopy == NULL) {
    mscclppGdrCopy = mscclppGdrInit();
    if (mscclppGdrCopy == NULL) {
      WARN("GDR init failed");
      return mscclppSystemError;
    }
  }
  return mscclppSuccess;
}

#endif

static mscclppResult_t mscclppInit()
{
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE))
    return mscclppSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    // Always initialize bootstrap network
    MSCCLPPCHECK(bootstrapNetInit());

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

MSCCLPP_API(mscclppResult_t, mscclppGetUniqueId, mscclppUniqueId* out);
mscclppResult_t mscclppGetUniqueId(mscclppUniqueId* out)
{
  MSCCLPPCHECK(mscclppInit());
  //   mscclppCHECK(PtrCheck(out, "GetUniqueId", "out"));
  mscclppResult_t res = bootstrapGetUniqueId((struct mscclppBootstrapHandle*)out);
  TRACE_CALL("mscclppGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}

MSCCLPP_API(mscclppResult_t, mscclppBootstrapAllGather, mscclppComm_t comm, void* data, int size);
mscclppResult_t mscclppBootstrapAllGather(mscclppComm_t comm, void* data, int size)
{
  MSCCLPPCHECK(bootstrapAllGather(comm->bootstrap, data, size));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppCommInitRank, mscclppComm_t* comm, int nranks, const char* ipPortPair, int rank);
mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, const char* ipPortPair, int rank)
{
#if defined(MSCCLPP_USE_GDRCOPY)
  MSCCLPPCHECK(initGdrCopy());
#endif

  mscclppResult_t res = mscclppSuccess;
  mscclppComm_t _comm = NULL;
  // uint64_t hash = getHostHash();
  // uint64_t *hashes;
  // std::map<uint64_t, int> hashToNode;

  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm, 1), res, fail);
  _comm->rank = rank;
  _comm->nRanks = nranks;
  _comm->devNumaNode = -1;
  // We assume that the user has set the device to the intended one already
  CUDACHECK(cudaGetDevice(&_comm->cudaDev));

  MSCCLPPCHECK(bootstrapNetInit(ipPortPair));
  mscclppBootstrapHandle handle;
  MSCCLPPCHECK(bootstrapGetUniqueId(&handle, rank == 0, ipPortPair));
  _comm->magic = handle.magic;

  MSCCLPPCHECKGOTO(mscclppCudaHostCalloc((uint32_t**)&_comm->abortFlag, 1), res, fail);
  MSCCLPPCHECK(bootstrapInit(&handle, _comm));

#if defined(ENABLE_NPKIT)
  // Init NPKit
  MSCCLPPCHECK(NpKit::Init(_comm->rank));
#endif

  *comm = _comm;
  return res;
fail:
  if (_comm) {
    if (_comm->abortFlag)
      mscclppCudaHostFree((void*)_comm->abortFlag);
    free(_comm);
  }
  if (comm)
    *comm = NULL;
  return res;
}

MSCCLPP_API(mscclppResult_t, mscclppCommInitRankFromId, mscclppComm_t* comm, int nranks, mscclppUniqueId id, int rank);
mscclppResult_t mscclppCommInitRankFromId(mscclppComm_t* comm, int nranks, mscclppUniqueId id, int rank)
{
#if defined(MSCCLPP_USE_GDRCOPY)
  MSCCLPPCHECK(initGdrCopy());
#endif

  mscclppResult_t res = mscclppSuccess;
  mscclppComm_t _comm = NULL;
  mscclppBootstrapHandle* handle = (mscclppBootstrapHandle*)&id;

  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm, 1), res, fail);
  _comm->rank = rank;
  _comm->nRanks = nranks;
  // We assume that the user has set the device to the intended one already
  CUDACHECK(cudaGetDevice(&_comm->cudaDev));

  MSCCLPPCHECK(bootstrapNetInit());
  _comm->magic = handle->magic;

  MSCCLPPCHECKGOTO(mscclppCudaHostCalloc((uint32_t**)&_comm->abortFlag, 1), res, fail);
  MSCCLPPCHECK(bootstrapInit(handle, _comm));

#if defined(ENABLE_NPKIT)
  // Init NPKit
  MSCCLPPCHECK(NpKit::Init(_comm->rank));
#endif

  *comm = _comm;
  return res;
fail:
  if (_comm) {
    if (_comm->abortFlag)
      mscclppCudaHostFree((void*)_comm->abortFlag);
    free(_comm);
  }
  if (comm)
    *comm = NULL;
  return res;
}

MSCCLPP_API(mscclppResult_t, mscclppCommDestroy, mscclppComm_t comm);
mscclppResult_t mscclppCommDestroy(mscclppComm_t comm)
{
#if defined(ENABLE_NPKIT)
  const char* npkitDumpDir = nullptr;
#endif

  if (comm == NULL)
    return mscclppSuccess;

  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    MSCCLPPCHECK(mscclppCudaFree(conn->devConn->proxyEpochId));
  }

  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState* proxyState = comm->proxyState[i];
    if (proxyState) {
#if defined(MSCCLPP_USE_GDRCOPY)
      MSCCLPPCHECK(mscclppGdrCudaFree(proxyState->triggerFifoDesc));
#else
      MSCCLPPCHECK(mscclppCudaHostFree(proxyState->triggerFifo));
#endif
      MSCCLPPCHECK(mscclppCudaFree(proxyState->fifoHead));
#if defined(MSCCLPP_USE_GDRCOPY)
      MSCCLPPCHECK(mscclppGdrCudaFree(proxyState->fifoTailDesc));
#else
      MSCCLPPCHECK(mscclppCudaFree(proxyState->fifoTailDev));
#endif
      if (proxyState->p2pStream)
        CUDACHECK(cudaStreamDestroy(proxyState->p2pStream));
      CUDACHECK(cudaStreamDestroy(proxyState->fifoStream));
      free(proxyState);
    }
  }

  for (int i = 0; i < MSCCLPP_IB_MAX_DEVS; ++i) {
    if (comm->ibContext[i]) {
      MSCCLPPCHECK(mscclppIbContextDestroy(comm->ibContext[i]));
    }
  }

  for (int i = 0; i < comm->nConns; i++) {
    struct mscclppConn* conn = &comm->conns[i];
    if (conn) {
      MSCCLPPCHECK(mscclppCudaFree(conn->devConn->sendEpochId));
      MSCCLPPCHECK(mscclppCudaFree(conn->devConn->recvEpochId));
    }
  }

  if (comm->bootstrap)
    MSCCLPPCHECK(bootstrapClose(comm->bootstrap));

  mscclppCudaHostFree((void*)comm->abortFlag);
  free(comm);

#if defined(ENABLE_NPKIT)
  // Dump NPKit events and shutdown
  npkitDumpDir = getenv("NPKIT_DUMP_DIR");
  if (npkitDumpDir == nullptr) {
    WARN("NPKIT_DUMP_DIR is empty");
  } else {
    MSCCLPPCHECK(NpKit::Dump(npkitDumpDir));
  }
  MSCCLPPCHECK(NpKit::Shutdown());
#endif

  return mscclppSuccess;
}

MSCCLPP_API(const char*, mscclppGetErrorString, mscclppResult_t code);
const char* mscclppGetErrorString(mscclppResult_t code)
{
  switch (code) {
  case mscclppSuccess:
    return "no error";
  case mscclppUnhandledCudaError:
    return "unhandled cuda error";
  case mscclppSystemError:
    return "unhandled system error";
  case mscclppInternalError:
    return "internal error";
  case mscclppInvalidArgument:
    return "invalid argument";
  case mscclppInvalidUsage:
    return "invalid usage";
  case mscclppRemoteError:
    return "remote process exited or there was a network error";
  case mscclppInProgress:
    return "MSCCL++ operation in progress";
  default:
    return "unknown result code";
  }
}

MSCCLPP_API(mscclppResult_t, mscclppGetDeviceConnection, mscclppComm_t comm, int remoteRank, int tag,
            mscclppDevConn_t** devConn);
mscclppResult_t mscclppGetDeviceConnection(mscclppComm_t comm, int remoteRank, int tag, mscclppDevConn_t** devConn)
{
  for (int i = 0; i < comm->nConns; i++) {
    if (comm->devConns[i].remoteRank == remoteRank && comm->devConns[i].tag == tag) {
      *devConn = &comm->devConns[i];
      return mscclppSuccess;
    }
  }

  return mscclppInvalidArgument;
}

MSCCLPP_API(mscclppResult_t, mscclppGetAllDeviceConnections, mscclppComm_t comm, mscclppDevConn_t** devConns,
            int* nConns);
mscclppResult_t mscclppGetAllDeviceConnections(mscclppComm_t comm, mscclppDevConn_t** devConns, int* nConns)
{
  *nConns = comm->nConns;
  *devConns = comm->devConns;
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppConnect, mscclppComm_t comm, int remoteRank, int tag, void* localBuff,
            uint64_t buffSize, mscclppTransport_t transportType, const char* ibDev);
mscclppResult_t mscclppConnect(mscclppComm_t comm, int remoteRank, int tag, void* localBuff, uint64_t buffSize,
                               mscclppTransport_t transportType, const char* ibDev)
{
  // save this processes numa binding and set it to the one closest to the device
  // so that all the allocation are close to the device
  if (comm->devNumaNode == -1) {
    // in case this is our first time
    MSCCLPPCHECK(getDeviceNumaNode(comm->cudaDev, &comm->devNumaNode));
    INFO(MSCCLPP_INIT, "NUMA node of device %d is set to %d", comm->cudaDev, comm->devNumaNode);
  }
  // save numa node bitmask to change it back to user's numa node
  mscclppNumaState curProcessState;
  MSCCLPPCHECK(getNumaState(&curProcessState));
  // change to device's numa node so that the following allocation are close to the device
  MSCCLPPCHECK(numaBind(comm->devNumaNode));

  if (comm->nConns == MAXCONNECTIONS) {
    WARN("Too many connections made");
    return mscclppInternalError;
  }
  struct mscclppConn* conn = &comm->conns[comm->nConns];
  conn->transport = transportType;
  conn->buffSize = buffSize;

  conn->ibCtx = NULL;
  conn->ibQp = NULL;
  int ibDevIdx = -1;
  if (transportType == mscclppTransportIB) {
    // Check if an IB context exists
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

    // If not, create a new one
    if (ibDevIdx == -1) {
      // Create a new context.
      ibDevIdx = firstNullIdx;
      if (mscclppIbContextCreate(&comm->ibContext[ibDevIdx], ibDev) != mscclppSuccess) {
        WARN("Failed to create IB context");
        return mscclppInternalError;
      }
    }
    // Set the ib context for this conn
    conn->ibCtx = comm->ibContext[ibDevIdx];
  } else if (transportType == mscclppTransportP2P) {
    // No allocation needed for P2P proxy
  } else if (transportType == mscclppTransportSHM) {
    WARN("Shared memory interconnection is not implemented yet!");
    return mscclppInternalError;
  } else {
    WARN("Unexpected connection type!");
    return mscclppInvalidUsage;
  }

  // Find/create a proxy state for the given connection
  struct mscclppProxyState* proxyState = NULL;
  // First see if there is a matching context
  // If not, find the first empty proxy
  int firstEmptyProxyIndex = -1;
  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState* curProxy = comm->proxyState[i];
    if (curProxy && (curProxy->transportType == transportType)) {
      if ((transportType == mscclppTransportIB && curProxy->ibContext == conn->ibCtx) ||
          (transportType == mscclppTransportP2P)) {
        proxyState = curProxy;
        break; // we found the matching context
      }
    }
    if (curProxy == NULL && firstEmptyProxyIndex == -1) {
      firstEmptyProxyIndex = i;
    }
  }

  if (proxyState == NULL && firstEmptyProxyIndex == -1) {
    WARN("Too many proxies have been allocated!");
    return mscclppInvalidUsage;
  }

  // If we couldn't find a matching context, create one
  if (proxyState == NULL) {
    MSCCLPPCHECK(mscclppCalloc(&proxyState, 1));
#if defined(MSCCLPP_USE_GDRCOPY)
    MSCCLPPCHECK(mscclppGdrCudaCalloc(&proxyState->triggerFifo, &proxyState->triggerFifoDev, MSCCLPP_PROXY_FIFO_SIZE,
                                      &proxyState->triggerFifoDesc));
#else
    MSCCLPPCHECK(mscclppCudaHostCalloc(&proxyState->triggerFifo, MSCCLPP_PROXY_FIFO_SIZE));
#endif
    MSCCLPPCHECK(mscclppCudaCalloc(&proxyState->fifoHead, 1));
#if defined(MSCCLPP_USE_GDRCOPY)
    MSCCLPPCHECK(
      mscclppGdrCudaCalloc(&proxyState->fifoTailDevHostPtr, &proxyState->fifoTailDev, 1, &proxyState->fifoTailDesc));
#else
    MSCCLPPCHECK(mscclppCudaCalloc(&proxyState->fifoTailDev, 1));
#endif
    proxyState->fifoTailHost = 0;

    if (transportType == mscclppTransportIB) {
      proxyState->ibContext = conn->ibCtx;
      proxyState->p2pStream = NULL;
    } else if (transportType == mscclppTransportP2P) {
      proxyState->ibContext = NULL;
      CUDACHECK(cudaStreamCreateWithFlags(&proxyState->p2pStream, cudaStreamNonBlocking));
    }
    CUDACHECK(cudaStreamCreateWithFlags(&proxyState->fifoStream, cudaStreamNonBlocking));
    proxyState->numaNodeToBind = comm->devNumaNode;

    // INFO(MSCCLPP_INIT, "NUMA node for device %d is %d", cudaDev, *numaNode);
    proxyState->transportType = transportType;
    comm->proxyState[firstEmptyProxyIndex] = proxyState;
  }
  if (proxyState == NULL) {
    // Cannot reach
    WARN("Proxy allocation failed!");
    return mscclppInternalError;
  }

  struct mscclppDevConn* devConn = &comm->devConns[comm->nConns];

  conn->devConn = devConn;
  conn->devConn->localBuff = localBuff;
  MSCCLPPCHECK(mscclppCudaCalloc(&conn->devConn->sendEpochId, 1));
  MSCCLPPCHECK(mscclppCudaCalloc(&conn->devConn->recvEpochId, 1));
  conn->devConn->remoteRank = remoteRank;
  conn->devConn->tag = tag;
  conn->devConn->fifo.connId = comm->nConns;
#if defined(MSCCLPP_USE_GDRCOPY)
  conn->devConn->fifo.triggerFifo = proxyState->triggerFifoDev;
#else
  conn->devConn->fifo.triggerFifo = proxyState->triggerFifo;
#endif
  conn->devConn->fifo.triggerFifoHead = proxyState->fifoHead;
  conn->devConn->fifo.triggerFifoTail = proxyState->fifoTailDev;

  comm->nConns++;

  // change the numa binding back to user's
  MSCCLPPCHECK(setNumaState(curProcessState));

  return mscclppSuccess;
}

struct connInfo
{
  cudaIpcMemHandle_t handleBuff;
  cudaIpcMemHandle_t handleFlag;
  cudaIpcMemHandle_t handleProxyFlag;
  mscclppIbQpInfo infoQp;
  mscclppIbMrInfo infoBuffMr;
  mscclppIbMrInfo infoLocalFlagMr;
  mscclppIbMrInfo infoProxyFlagMr;
};

mscclppResult_t mscclppP2pConnectionSetupStart(struct connInfo* connInfo /*output*/, struct mscclppConn* conn /*input*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("connInfo or connection cannot be null");
    return mscclppInternalError;
  }
  struct mscclppDevConn* devConn = conn->devConn;
  MSCCLPPCHECK(mscclppCudaCalloc(&devConn->proxyEpochId, 1));
  CUDACHECK(cudaIpcGetMemHandle(&connInfo->handleProxyFlag, devConn->proxyEpochId));
  CUDACHECK(cudaIpcGetMemHandle(&connInfo->handleBuff, devConn->localBuff));
  CUDACHECK(cudaIpcGetMemHandle(&connInfo->handleFlag, devConn->sendEpochId));
  return mscclppSuccess;
}

mscclppResult_t mscclppP2pConnectionSetupEnd(struct connInfo* connInfo /*input*/, struct mscclppConn* conn /*output*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("ipcHandles or connection cannot be null");
    return mscclppInternalError;
  }
  CUDACHECK(
    cudaIpcOpenMemHandle((void**)&conn->devConn->remoteBuff, connInfo->handleBuff, cudaIpcMemLazyEnablePeerAccess));
  CUDACHECK(
    cudaIpcOpenMemHandle((void**)&conn->devConn->remoteFlag, connInfo->handleFlag, cudaIpcMemLazyEnablePeerAccess));
  CUDACHECK(
    cudaIpcOpenMemHandle((void**)&conn->remoteProxyFlag, connInfo->handleProxyFlag, cudaIpcMemLazyEnablePeerAccess));
  return mscclppSuccess;
}

mscclppResult_t mscclppIbConnectionSetupStart(struct connInfo* connInfo /*output*/, struct mscclppConn* conn /*input*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("connInfo or connection cannot be null");
    return mscclppInternalError;
  }
  struct mscclppDevConn* devConn = conn->devConn;
  devConn->remoteBuff = NULL;
  devConn->remoteFlag = NULL;
  MSCCLPPCHECK(mscclppCudaCalloc(&devConn->proxyEpochId, 1));

  struct mscclppIbContext* ibCtx = conn->ibCtx;
  if (conn->ibQp == NULL) {
    MSCCLPPCHECK(mscclppIbContextCreateQp(ibCtx, &conn->ibQp));
  }
  // TODO(chhwang): can we register only one MR for the following three?
  MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->localBuff, conn->buffSize, &conn->ibBuffMr));
  MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->sendEpochId, sizeof(uint64_t), &conn->ibLocalFlagMr));
  MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, devConn->proxyEpochId, sizeof(uint64_t), &conn->ibProxyFlagMr));
  connInfo->infoQp = conn->ibQp->info;
  connInfo->infoBuffMr = conn->ibBuffMr->info;
  connInfo->infoLocalFlagMr = conn->ibLocalFlagMr->info;
  connInfo->infoProxyFlagMr = conn->ibProxyFlagMr->info;
  return mscclppSuccess;
}

mscclppResult_t mscclppIbConnectionSetupEnd(struct connInfo* connInfo /*input*/, struct mscclppConn* conn /*output*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("ipcHandles or connection cannot be null");
    return mscclppInternalError;
  }
  if (conn->ibQp->rtr(&connInfo->infoQp) != 0) {
    WARN("Failed to transition QP to RTR");
    return mscclppInvalidUsage;
  }
  if (conn->ibQp->rts() != 0) {
    WARN("Failed to transition QP to RTS");
    return mscclppInvalidUsage;
  }
  conn->ibBuffMrInfo = connInfo->infoBuffMr;
  conn->ibLocalFlagMrInfo = connInfo->infoLocalFlagMr;
  conn->ibProxyFlagMrInfo = connInfo->infoProxyFlagMr;
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppConnectionSetup, mscclppComm_t comm);
mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm)
{
  // Send info to peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];

    struct connInfo cInfo;
    if (conn->transport == mscclppTransportP2P) {
      MSCCLPPCHECK(mscclppP2pConnectionSetupStart(&cInfo, conn));
    } else if (conn->transport == mscclppTransportIB) {
      MSCCLPPCHECK(mscclppIbConnectionSetupStart(&cInfo, conn));
    }
    // TODO: from saemal: do we possibly deadlock if there are too many outstanding sends?
    MSCCLPPCHECK(bootstrapSend(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag, &cInfo, sizeof(cInfo)));
  }

  // Recv info from peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    struct connInfo cInfo;
    MSCCLPPCHECK(bootstrapRecv(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag, &cInfo, sizeof(cInfo)));
    if (conn->transport == mscclppTransportP2P) {
      MSCCLPPCHECK(mscclppP2pConnectionSetupEnd(&cInfo, conn));
    } else if (conn->transport == mscclppTransportIB) {
      MSCCLPPCHECK(mscclppIbConnectionSetupEnd(&cInfo, conn));
    }
  }

  // a barrier to ensure setup on all gpus are done and we can return to the user
  MSCCLPPCHECK(mscclppBootstrapBarrier(comm));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppProxyLaunch, mscclppComm_t comm);
mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm)
{
  MSCCLPPCHECK(mscclppProxyCreate(comm));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppBootstrapBarrier, mscclppComm_t comm);
mscclppResult_t mscclppBootstrapBarrier(mscclppComm_t comm)
{
  int* tmp = new int[comm->nRanks];
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
  delete[] tmp;
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppProxyStop, mscclppComm_t comm);
mscclppResult_t mscclppProxyStop(mscclppComm_t comm)
{
  // a barrier to make sure all ranks are done with their work before stopping the proxy
  MSCCLPPCHECK(mscclppBootstrapBarrier(comm));

  MSCCLPPCHECK(mscclppProxyDestroy(comm));
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppCommRank, mscclppComm_t comm, int* rank);
mscclppResult_t mscclppCommRank(mscclppComm_t comm, int* rank)
{
  if (comm == NULL || rank == NULL) {
    WARN("comm or rank cannot be null");
    return mscclppInvalidUsage;
  }
  *rank = comm->rank;
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppCommSize, mscclppComm_t comm, int* size);
mscclppResult_t mscclppCommSize(mscclppComm_t comm, int* size)
{
  if (comm == NULL || size == NULL) {
    WARN("comm or size cannot be null");
    return mscclppInvalidUsage;
  }
  *size = comm->nRanks;
  return mscclppSuccess;
}

MSCCLPP_API(void, mscclppDefaultLogHandler, const char* msg);
void mscclppDefaultLogHandler(const char* msg)
{
  mscclppDebugDefaultLogHandler(msg);
}

MSCCLPP_API(mscclppResult_t, mscclppSetLogHandler, mscclppLogHandler_t handler);
mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler)
{
  return mscclppDebugSetLogHandler(handler);
}

MSCCLPP_API(mscclppResult_t, mscclppSetBootstrapConnTimeout, int timeout);
mscclppResult_t mscclppSetBootstrapConnTimeout(int timeout)
{
  mscclppConfig* config = mscclppConfig::getInstance();
  config->setBootstrapConnectionTimeoutConfig(timeout);
  return mscclppSuccess;
}