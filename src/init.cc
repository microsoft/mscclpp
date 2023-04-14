#include "alloc.h"
#include "api.h"
#include "bootstrap.h"
#include "checks.h"
#include "config.h"
#if defined(MSCCLPP_USE_GDRCOPY)
#include "gdr.h"
#endif
#include "mscclpp.h"
#include <map>
#include <sstream>
#include <vector>
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

MSCCLPP_API mscclppResult_t mscclppGetUniqueId(mscclppUniqueId* out)
{
  MSCCLPPCHECK(mscclppInit());
  //   mscclppCHECK(PtrCheck(out, "GetUniqueId", "out"));
  mscclppResult_t res = bootstrapGetUniqueId((struct mscclppBootstrapHandle*)out);
  TRACE_CALL("mscclppGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}

MSCCLPP_API mscclppResult_t mscclppBootstrapAllGather(mscclppComm_t comm, void* data, int size)
{
  MSCCLPPCHECK(bootstrapAllGather(comm->bootstrap, data, size));
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppCommInitRank(mscclppComm_t* comm, int nranks, const char* ipPortPair, int rank)
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

MSCCLPP_API mscclppResult_t mscclppCommInitRankFromId(mscclppComm_t* comm, int nranks, mscclppUniqueId id, int rank)
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

MSCCLPP_API mscclppResult_t mscclppCommDestroy(mscclppComm_t comm)
{
#if defined(ENABLE_NPKIT)
  const char* npkitDumpDir = nullptr;
#endif

  if (comm == NULL)
    return mscclppSuccess;

  for (int i = 0; i < MSCCLPP_PROXY_MAX_NUM; ++i) {
    struct mscclppProxyState* proxyState = comm->proxyState[i];
    if (proxyState) {
      MSCCLPPCHECK(proxyState->fifo.destroy());
      if (proxyState->p2pStream)
        CUDACHECK(cudaStreamDestroy(proxyState->p2pStream));
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
      MSCCLPPCHECK(mscclppCudaFree(conn->devConn->localSignalEpochId));
      MSCCLPPCHECK(mscclppCudaFree(conn->devConn->waitEpochId));
      if (conn->hostConn)
        delete conn->hostConn;
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

MSCCLPP_API const char* mscclppGetErrorString(mscclppResult_t code)
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

MSCCLPP_API mscclppResult_t mscclppGetDeviceConnection(mscclppComm_t comm, int remoteRank, int tag,
                                                       mscclppDevConn_t** devConn)
{
  for (int i = 0; i < comm->nConns; i++) {
    if (comm->devConns[i].remoteRank == remoteRank && comm->devConns[i].tag == tag) {
      *devConn = &comm->devConns[i];
      return mscclppSuccess;
    }
  }

  return mscclppInvalidArgument;
}

MSCCLPP_API mscclppResult_t mscclppGetAllDeviceConnections(mscclppComm_t comm, mscclppDevConn_t** devConns, int* nConns)
{
  *nConns = comm->nConns;
  *devConns = comm->devConns;
  return mscclppSuccess;
}

#if defined(ENABLE_NPKIT)

static void npkitInitReqIds(struct mscclppComm* comm)
{
  for (int i = 0; i < comm->nConns; i++) {
    struct mscclppConn* conn = &comm->conns[i];
    conn->npkitUsedReqIds.resize(0);
    conn->npkitFreeReqIds.resize(MSCCLPP_IB_MAX_SENDS);
    for (uint64_t j = 0; j < MSCCLPP_IB_MAX_SENDS; j++) {
      conn->npkitFreeReqIds[j] = MSCCLPP_IB_MAX_SENDS - j - 1;
    }
  }
}

static void npkitCollectEntryEvent(struct mscclppConn* conn, uint8_t type, uint32_t size)
{
  uint64_t reqId = 0;
  if (conn->npkitFreeReqIds.size() == 0) {
    reqId = conn->npkitUsedReqIds.size();
  } else {
    reqId = conn->npkitFreeReqIds.back();
    conn->npkitFreeReqIds.pop_back();
  }
  conn->npkitUsedReqIds.push_back(reqId);
  NpKit::CollectCpuEvent(type, size, (uint32_t)reqId, NpKit::GetCpuTimestamp(), conn->connId);
}

static void npkitCollectExitEvents(struct mscclppConn* conn, uint8_t type)
{
  while (conn->npkitUsedReqIds.size()) {
    uint64_t reqId = conn->npkitUsedReqIds.back();
    NpKit::CollectCpuEvent(type, 0, (uint32_t)reqId, NpKit::GetCpuTimestamp(), conn->connId);
    conn->npkitFreeReqIds.push_back(reqId);
    conn->npkitUsedReqIds.pop_back();
  }
}

#else

#define npkitInitReqIds(comm)

#define npkitCollectEntryEvent(conn, type, size)

#define npkitCollectExitEvents(conn, type)

#endif

struct mscclppHostP2PConn : mscclppHostConn
{
  mscclppHostP2PConn(mscclppConn* _conn, cudaStream_t _stream) : conn(_conn), p2pStream(_stream)
  {
  }

  void put(uint64_t dstDataOffset, uint64_t srcDataOffset, uint64_t dataSize)
  {
    put(1, dstDataOffset, 1, srcDataOffset, dataSize);
  }
  void put(mscclppBufferHandle_t dst, uint64_t dstDataOffset, mscclppBufferHandle_t src, uint64_t srcDataOffset, uint64_t dataSize)
  {
    void* srcBuff = (void*)((char*)conn->bufferRegistrations[src].data + srcDataOffset);
    void* dstBuff = (void*)((char*)conn->remoteBufferRegistrations[dst].data + dstDataOffset);
    CUDACHECKNORET(cudaMemcpyAsync(dstBuff, srcBuff, dataSize, cudaMemcpyDeviceToDevice, p2pStream));
    npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)dataSize);
  }
  void signal()
  {
    CUDACHECKNORET(cudaMemcpyAsync(&conn->devConn->remoteSignalEpochId->proxy,
                                   &(conn->devConn->localSignalEpochId->device), sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice, p2pStream));
    npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_FLAG_ENTRY, (uint32_t)sizeof(uint64_t));
  }
  void wait()
  {
  }
  void flush()
  {
    CUDACHECKNORET(cudaStreamSynchronize(p2pStream));
    npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
  }

  mscclppConn* conn;
  cudaStream_t p2pStream;
};

struct mscclppHostIBConn : mscclppHostConn
{
  mscclppHostIBConn(mscclppConn* conn) : conn(conn)
  {
    this->ibQp = NULL;
  }

  void put(uint64_t dstDataOffset, uint64_t srcDataOffset, uint64_t dataSize)
  {
    put(1, dstDataOffset, 1, srcDataOffset, dataSize);
  }
  void put(mscclppBufferHandle_t dst, uint64_t dstDataOffset, mscclppBufferHandle_t src, uint64_t srcDataOffset, uint64_t dataSize)
  {
    this->ibQp->stageSend(this->ibMrs[src], &this->remoteIbMrInfos[dst], (uint32_t)dataSize,
                          /*wrId=*/0, /*srcOffset=*/srcDataOffset, /*dstOffset=*/dstDataOffset, /*signaled=*/false);
    int ret = this->ibQp->postSend();
    if (ret != 0) {
      // Return value is errno.
      WARN("data postSend failed: errno %d", ret);
    }
    npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)dataSize);
  }
  void signal()
  {
    // My local device flag is copied to the remote's proxy flag
    this->ibQp->stageSend(this->ibMrs[0], &this->remoteIbMrInfos[0], sizeof(uint64_t),
                          /*wrId=*/0, /*srcOffset=*/0, /*dstOffset=*/sizeof(uint64_t), /*signaled=*/true);
    int ret = this->ibQp->postSend();
    if (ret != 0) {
      WARN("flag postSend failed: errno %d", ret);
    }
    npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_FLAG_ENTRY, (uint32_t)sizeof(uint64_t));
  }
  void wait()
  {
  }
  void flush()
  {
    bool isWaiting = true;
    while (isWaiting) {
      int wcNum = this->ibQp->pollCq();
      if (wcNum < 0) {
        WARN("pollCq failed: errno %d", errno);
        continue;
      }
      for (int i = 0; i < wcNum; ++i) {
        struct ibv_wc* wc = &this->ibQp->wcs[i];
        if (wc->status != IBV_WC_SUCCESS) {
          WARN("wc status %d", wc->status);
          continue;
        }
        if (wc->qp_num != this->ibQp->qp->qp_num) {
          WARN("got wc of unknown qp_num %d", wc->qp_num);
          continue;
        }
        if (wc->opcode == IBV_WC_RDMA_WRITE) {
          isWaiting = false;
          break;
        }
      }
    }
    npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
  }

  mscclppConn* conn;
  struct mscclppIbQp* ibQp;
  std::vector<mscclppIbMr*> ibMrs;
  std::vector<mscclppIbMrInfo> remoteIbMrInfos;
};

MSCCLPP_API mscclppResult_t mscclppConnectWithoutBuffer(mscclppComm_t comm, int remoteRank, int tag, mscclppTransport_t transportType, const char* ibDev)
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
  int connId = comm->nConns;
  struct mscclppConn* conn = &comm->conns[connId];
  conn->connId = connId;
  conn->transport = transportType;
  conn->buffSize = 0;

  conn->ibCtx = NULL;
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
    // do the rest of the initialization later
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
    MSCCLPPCHECK(proxyState->fifo.create());

    if (transportType == mscclppTransportIB) {
      proxyState->ibContext = conn->ibCtx;
      proxyState->p2pStream = NULL;
    } else if (transportType == mscclppTransportP2P) {
      proxyState->ibContext = NULL;
      CUDACHECK(cudaStreamCreateWithFlags(&proxyState->p2pStream, cudaStreamNonBlocking));
    }
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

  if (transportType == mscclppTransportIB) {
    conn->hostConn = new mscclppHostIBConn(conn);
  } else if (transportType == mscclppTransportP2P) {
    conn->hostConn = new mscclppHostP2PConn(conn, proxyState->p2pStream);
  }

  struct mscclppDevConn* devConn = &comm->devConns[connId];

  conn->devConn = devConn;
  conn->devConn->localBuff = nullptr;
  MSCCLPPCHECK(mscclppCudaCalloc(&conn->devConn->localSignalEpochId, 1));
  MSCCLPPCHECK(mscclppCudaCalloc(&conn->devConn->waitEpochId, 1));
  conn->devConn->remoteRank = remoteRank;
  conn->devConn->tag = tag;
  conn->devConn->fifo.connId = connId;
#if defined(MSCCLPP_USE_GDRCOPY)
  conn->devConn->fifo.triggerFifo = proxyState->fifo.triggerFifoDev;
#else
  conn->devConn->fifo.triggerFifo = proxyState->fifo.triggerFifo;
#endif
  conn->devConn->fifo.triggerFifoHead = proxyState->fifo.fifoHead;
  conn->devConn->fifo.triggerFifoTail = proxyState->fifo.fifoTailDev;

  comm->nConns++;

  // change the numa binding back to user's
  MSCCLPPCHECK(setNumaState(curProcessState));

  mscclppBufferHandle_t signalHandle = -1;
  MSCCLPPCHECK(mscclppRegisterBufferForConnection(comm, connId, conn->devConn->localSignalEpochId, sizeof(mscclppDevConnSignalEpochId), &signalHandle));
  if (signalHandle != 0) {
    WARN("signal handle should be 0");
    return mscclppInternalError;
  }

  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppConnect(mscclppComm_t comm, int remoteRank, int tag, void* localBuff,
                                           uint64_t buffSize, mscclppTransport_t transportType, const char* ibDev)
{
  int connId = comm->nConns;
  MSCCLPPCHECK(mscclppConnectWithoutBuffer(comm, remoteRank, tag, transportType, ibDev));
  struct mscclppConn* conn = &comm->conns[connId];

  conn->buffSize = buffSize;
  conn->devConn->localBuff = localBuff;

  mscclppBufferHandle_t localBuffHandle = -1;
  MSCCLPPCHECK(mscclppRegisterBufferForConnection(comm, connId, conn->devConn->localSignalEpochId, buffSize, &localBuffHandle));
  if (localBuffHandle != 1) {
    WARN("data buffer handle should be 1");
    return mscclppInternalError;
  }

  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppRegisterBufferForConnection(mscclppComm_t comm, int connIdx, void* localBuff, uint64_t buffSize, mscclppBufferHandle_t *handle) {
  if (connIdx >= comm->nConns) {
    WARN("connIdx out of range");
    return mscclppInvalidArgument;
  }
  mscclppConn& conn = comm->conns[connIdx];
  *handle = conn.bufferRegistrations.size();
  conn.bufferRegistrations.emplace_back();
  conn.bufferRegistrations.back().data = localBuff;
  conn.bufferRegistrations.back().size = buffSize;

  return mscclppSuccess;
}

struct mscclppBufferRegistrationInfo
{
  cudaIpcMemHandle_t cudaHandle;
  mscclppIbMrInfo ibMrInfo;
  uint64_t size;
};

struct connInfo
{
  mscclppIbQpInfo infoQp;
  std::vector<mscclppBufferRegistrationInfo> bufferInfos;

  struct header {
    mscclppIbQpInfo infoQp;
    int numBufferInfos;
  };

  mscclppResult_t sendOverBootstrap(void* bootstrap, int remoteRank, int tag) {
    header h;
    h.infoQp = infoQp;
    h.numBufferInfos = bufferInfos.size();
    MSCCLPPCHECK(bootstrapSend(bootstrap, remoteRank, tag, &h, sizeof(header)));
    MSCCLPPCHECK(bootstrapSend(bootstrap, remoteRank, tag, bufferInfos.data(), bufferInfos.size() * sizeof(mscclppBufferRegistrationInfo)));
  return mscclppSuccess;
  }

  mscclppResult_t recvOverBootstrap(void* bootstrap, int remoteRank, int tag) {
    header h;
    MSCCLPPCHECK(bootstrapRecv(bootstrap, remoteRank, tag, &h, sizeof(header)));
    infoQp = h.infoQp;
    bufferInfos.resize(h.numBufferInfos);
    MSCCLPPCHECK(bootstrapRecv(bootstrap, remoteRank, tag, bufferInfos.data(), bufferInfos.size() * sizeof(mscclppBufferRegistrationInfo)));
  return mscclppSuccess;
  }
};

mscclppResult_t mscclppP2pConnectionSetupStart(struct connInfo* connInfo /*input*/, struct mscclppConn* conn /*input*/)
{
  if (conn == NULL) {
    WARN("connection cannot be null");
    return mscclppInternalError;
  }

  // Add all registered buffers
  for (const auto &bufReg : conn->bufferRegistrations) {
    connInfo->bufferInfos.emplace_back();
    CUDACHECK(cudaIpcGetMemHandle(&connInfo->bufferInfos.back().cudaHandle, bufReg.data));
    connInfo->bufferInfos.back().size = bufReg.size;
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppP2pConnectionSetupEnd(struct connInfo* connInfo /*input*/, struct mscclppConn* conn /*output*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("ipcHandles or connection cannot be null");
    return mscclppInternalError;
  }
  if (connInfo->bufferInfos.size() < 1) {
    WARN("at least 1 buffer info expected");
    return mscclppInternalError;
  }

  // Open all remote registered buffers
  for (size_t i = 0; i < connInfo->bufferInfos.size(); i++) {
    mscclppBufferRegistration newBufReg;
    CUDACHECK(cudaIpcOpenMemHandle(&newBufReg.data, connInfo->bufferInfos[i].cudaHandle, cudaIpcMemLazyEnablePeerAccess));
    newBufReg.size = connInfo->bufferInfos[i].size;
    conn->remoteBufferRegistrations.push_back(newBufReg);
  }

  if (conn->remoteBufferRegistrations[0].size != sizeof(mscclppDevConnSignalEpochId)) {
    WARN("buffer registration zero size doesn't match sizeof(mscclppDevConnSignalEpochId)");
    return mscclppInternalError;
  }
  conn->devConn->remoteSignalEpochId = (mscclppDevConnSignalEpochId*)conn->remoteBufferRegistrations[0].data;

  // For backwards compatibility with the previous API that assumed one data buffer per connection, set the remote buffer
  // to the first remote data buffer
  if (conn->remoteBufferRegistrations.size() > 1) {
    conn->devConn->remoteBuff = conn->remoteBufferRegistrations[1].data;
  }
  return mscclppSuccess;
}

mscclppResult_t mscclppIbConnectionSetupStart(struct connInfo* connInfo /*output*/, struct mscclppConn* conn /*input*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("connInfo or connection cannot be null");
    return mscclppInternalError;
  }
  struct mscclppDevConn* devConn = conn->devConn;
  struct mscclppHostIBConn* hostConn = (struct mscclppHostIBConn*)conn->hostConn;
  devConn->remoteBuff = NULL;
  devConn->remoteSignalEpochId = NULL;

  struct mscclppIbContext* ibCtx = conn->ibCtx;
  if (hostConn->ibQp == NULL) {
    MSCCLPPCHECK(mscclppIbContextCreateQp(ibCtx, &hostConn->ibQp));
  }

  // Add all registered buffers
  for (const auto &bufReg : conn->bufferRegistrations) {
    hostConn->ibMrs.emplace_back();
    MSCCLPPCHECK(mscclppIbContextRegisterMr(ibCtx, bufReg.data,
                                            sizeof(struct mscclppDevConnSignalEpochId), &hostConn->ibMrs.back()));
    connInfo->bufferInfos.emplace_back();
    connInfo->bufferInfos.back().ibMrInfo = hostConn->ibMrs.back()->info;
    connInfo->bufferInfos.back().size = bufReg.size;
  }

  connInfo->infoQp = hostConn->ibQp->info;
  return mscclppSuccess;
}

mscclppResult_t mscclppIbConnectionSetupEnd(struct connInfo* connInfo /*input*/, struct mscclppConn* conn /*output*/)
{
  if (connInfo == NULL || conn == NULL) {
    WARN("ipcHandles or connection cannot be null");
    return mscclppInternalError;
  }
  struct mscclppHostIBConn* hostConn = (struct mscclppHostIBConn*)conn->hostConn;
  if (hostConn->ibQp->rtr(&connInfo->infoQp) != 0) {
    WARN("Failed to transition QP to RTR");
    return mscclppInvalidUsage;
  }
  if (hostConn->ibQp->rts() != 0) {
    WARN("Failed to transition QP to RTS");
    return mscclppInvalidUsage;
  }

  // No remote pointers to set with IB, so we just set the Mrs

  // Push the Mrs for all the remote registered buffers
  for (size_t i = 1; i < connInfo->bufferInfos.size(); i++) {
    hostConn->remoteIbMrInfos.push_back(connInfo->bufferInfos[i].ibMrInfo);

    mscclppBufferRegistration newBufReg;
    newBufReg.data = nullptr;
    newBufReg.size = connInfo->bufferInfos[i].size;
    conn->remoteBufferRegistrations.push_back(newBufReg);
  }
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm)
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
    // MSCCLPPCHECK(bootstrapSend(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag, &cInfo, sizeof(cInfo)));
    MSCCLPPCHECK(cInfo.sendOverBootstrap(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag));
  }

  // Recv info from peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    struct connInfo cInfo;
    MSCCLPPCHECK(cInfo.recvOverBootstrap(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag));
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

struct bufferInfo
{
  cudaIpcMemHandle_t handleBuff;
  mscclppIbMrInfo infoBuffMr;
};

MSCCLPP_API mscclppResult_t mscclppRegisterBuffer(mscclppComm_t comm, void* local_memory, size_t size,
                                                  mscclppRegisteredMemory* regMem)
{
  std::vector<struct mscclppIbMr*> ibMrs;
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    struct bufferInfo bInfo;
    struct mscclppIbMr* ibBuffMr;

    // TODO: (conn->transport & mscclppTransportP2P) to support both P2P and IB
    if (conn->transport == mscclppTransportP2P) {
      CUDACHECK(cudaIpcGetMemHandle(&bInfo.handleBuff, local_memory));
    } else if (conn->transport == mscclppTransportIB) {
      MSCCLPPCHECK(mscclppIbContextRegisterMr(conn->ibCtx, local_memory, size, &ibBuffMr));
      bInfo.infoBuffMr = ibBuffMr->info;
      ibMrs.push_back(ibBuffMr);
    }

    MSCCLPPCHECK(bootstrapSend(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag, &bInfo, sizeof(bInfo)));
  }

  // Recv info from peers
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    struct bufferInfo bInfo;

    mscclppRegisteredMemoryP2P p2p;
    p2p.IbMr = NULL;
    p2p.remoteBuff = NULL;
    MSCCLPPCHECK(bootstrapRecv(comm->bootstrap, conn->devConn->remoteRank, conn->devConn->tag, &bInfo, sizeof(bInfo)));

    // TODO: (conn->transport & mscclppTransportP2P) to support both P2P and IB
    if (conn->transport == mscclppTransportP2P) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&p2p.remoteBuff, bInfo.handleBuff, cudaIpcMemLazyEnablePeerAccess));
    } else if (conn->transport == mscclppTransportIB) {
      p2p.IbMr = ibMrs[i];
    }
    regMem->p2p.push_back(p2p);
  }
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppRegisteredBufferWrite(mscclppComm_t comm, mscclppRegisteredMemory* regMem,
                                                         void* srcBuff, size_t size, uint32_t srcOffset,
                                                         uint32_t dstOffset, int64_t stream)
{
  int ret = 0;
  // TODO: transport should be an argument too so user can decide which transport to use
  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn* conn = &comm->conns[i];
    // TODO: (conn->transport & mscclppTransportP2P) to support both P2P and IB
    if (conn->transport == mscclppTransportP2P) {
      void* dstBuff = regMem->p2p[i].remoteBuff;
      CUDACHECK(cudaMemcpyAsync(dstBuff, srcBuff, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    } else {
      WARN("mscclppRegisteredBufferWrite not implemented for IB");
      return mscclppInternalError;
      // TODO: fix the following (Olli: probably by including the relevant ibBuffMr in the mscclppRegisteredMemory)
      // struct mscclppHostIBConn* hostConn = (struct mscclppHostIBConn*)conn->hostConn;
      // hostConn->ibQp->stageSend(hostConn->ibBuffMr, &hostConn->ibBuffMrRemoteInfo, (uint32_t)size,
      //                           /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset, /*signaled=*/false);
      // if ((ret = hostConn->ibQp->postSend()) != 0) {
      //   // Return value is errno.
      //   WARN("data postSend failed: errno %d", ret);
      // }
      // // ??
      // // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_ENTRY, (uint32_t)trigger.fields.dataSize,
      // // trigger.fields.connId);
    }
  }
  return mscclppSuccess;
}

// TODO: destroy registered buffer

MSCCLPP_API mscclppResult_t mscclppProxyLaunch(mscclppComm_t comm)
{
  npkitInitReqIds(comm);
  MSCCLPPCHECK(mscclppProxyCreate(comm));
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppBootstrapBarrier(mscclppComm_t comm)
{
  int* tmp = new int[comm->nRanks];
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
  delete[] tmp;
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppProxyStop(mscclppComm_t comm)
{
  // a barrier to make sure all ranks are done with their work before stopping the proxy
  MSCCLPPCHECK(mscclppBootstrapBarrier(comm));

  MSCCLPPCHECK(mscclppProxyDestroy(comm));
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppCommRank(mscclppComm_t comm, int* rank)
{
  if (comm == NULL || rank == NULL) {
    WARN("comm or rank cannot be null");
    return mscclppInvalidUsage;
  }
  *rank = comm->rank;
  return mscclppSuccess;
}

MSCCLPP_API mscclppResult_t mscclppCommSize(mscclppComm_t comm, int* size)
{
  if (comm == NULL || size == NULL) {
    WARN("comm or size cannot be null");
    return mscclppInvalidUsage;
  }
  *size = comm->nRanks;
  return mscclppSuccess;
}

MSCCLPP_API void mscclppDefaultLogHandler(const char* msg)
{
  mscclppDebugDefaultLogHandler(msg);
}

MSCCLPP_API mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler)
{
  return mscclppDebugSetLogHandler(handler);
}

MSCCLPP_API mscclppResult_t mscclppSetBootstrapConnTimeout(int timeout)
{
  mscclppConfig* config = mscclppConfig::getInstance();
  config->setBootstrapConnectionTimeoutConfig(timeout);
  return mscclppSuccess;
}
