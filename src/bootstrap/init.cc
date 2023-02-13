#include "mscclpp.h"
#include "bootstrap.h"
#include "core.h"
#include "shmutils.h"
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

  if (comm->bootstrap)
    MSCCLPPCHECK(bootstrapClose(comm->bootstrap));

  mscclppCudaHostFree((void *)comm->abortFlag);
  free(comm);
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppConnect, mscclppComm_t comm, int rankRecv, int rankSend,
            void *buff, int *flag, int tag, mscclppTransport_t transportType, const char *ibDev);
mscclppResult_t mscclppConnect(mscclppComm_t comm, int rankRecv, int rankSend, void *buff, int *flag, int tag,
                               mscclppTransport_t transportType, const char *ibDev/*=NULL*/)
{
  if (comm->rank == rankRecv || comm->rank == rankSend) {
    struct mscclppConn *conn = &comm->conns[comm->nConns++];
    conn->transport = transportType;
    conn->localRank = comm->rank;
    conn->ibDev = ibDev;
    conn->tag = tag;
    conn->buff = buff;
    conn->flag = flag;
    conn->remoteRank = (comm->rank == rankRecv) ? rankSend : rankRecv;
  }
  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppConnectionSetup, mscclppComm_t comm);
mscclppResult_t mscclppConnectionSetup(mscclppComm_t comm)
{
  struct ipcMemHandleInfo {
    cudaIpcMemHandle_t handle_buff;
    cudaIpcMemHandle_t handle_flag;
    int tag;
    int valid;
  };

  size_t shmSize = MAXCONNECTIONS * sizeof(struct ipcMemHandleInfo);
  int fd;
  struct ipcMemHandleInfo *handleInfos;
  std::string shmname = mscclppShmFileName(comm, comm->localRank);
  MSCCLPPCHECK(mscclppShmutilsMapCreate(shmname.c_str(), shmSize, &fd, (void **)&handleInfos));

  for (int i = 0; i < comm->nConns; ++i) {
    struct mscclppConn *conn = &comm->conns[i];
    CUDACHECK(cudaIpcGetMemHandle(&handleInfos[i].handle_buff, conn->buff));
    CUDACHECK(cudaIpcGetMemHandle(&handleInfos[i].handle_flag, conn->flag));
    handleInfos[i].tag = conn->tag;
    handleInfos[i].valid = 1;
  }

  // Local intra-node barrier: wait for all local ranks to have written their memory handles
  MSCCLPPCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  MSCCLPPCHECK(mscclppCudaHostCalloc(&comm->devConns, comm->nConns));

  for (int r = 0; r < comm->localRanks; ++r) {
    if (r == comm->localRank)
      continue;
    int fd_r;
    struct ipcMemHandleInfo *handleInfos_r;
    std::string shmname_r = mscclppShmFileName(comm, r);
    MSCCLPPCHECK(mscclppShmutilsMapOpen(shmname_r.c_str(), shmSize, &fd_r, (void **)&handleInfos_r));

    std::map<int, std::pair<cudaIpcMemHandle_t, cudaIpcMemHandle_t>> remoteHandles;
    for (int i = 0; i < MAXCONNECTIONS; ++i) {
      if (handleInfos_r[i].valid != 1) {
        break;
      }
      remoteHandles[handleInfos_r[i].tag] = std::make_pair(handleInfos_r[i].handle_buff, handleInfos_r[i].handle_flag);
    }

    for (int i = 0; i < comm->nConns; ++i) {
      struct mscclppConn *conn = &comm->conns[i];
      auto it = remoteHandles.find(conn->tag);
      if (it != remoteHandles.end()) {
        comm->devConns[i].tag = conn->tag;
        comm->devConns[i].localBuff = conn->buff;
        comm->devConns[i].localFlag = conn->flag;
        CUDACHECK(cudaIpcOpenMemHandle(&comm->devConns[i].remoteBuff, it->second.first, cudaIpcMemLazyEnablePeerAccess));
        CUDACHECK(cudaIpcOpenMemHandle((void **)&comm->devConns[i].remoteFlag, it->second.second, cudaIpcMemLazyEnablePeerAccess));
      }
    }

    MSCCLPPCHECK(mscclppShmutilsMapClose(shmname_r.c_str(), shmSize, fd_r, handleInfos_r));
  }

  // Local intra-node barrier: wait for all local ranks to have read all memory handles
  MSCCLPPCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  MSCCLPPCHECK(mscclppShmutilsMapDestroy(shmname.c_str(), shmSize, fd, handleInfos));

  return mscclppSuccess;
}

MSCCLPP_API(mscclppResult_t, mscclppGetDevConns, mscclppComm_t comm, mscclppDevConn_t* devConns);
mscclppResult_t mscclppGetDevConns(mscclppComm_t comm, mscclppDevConn_t* devConns)
{
  *devConns = comm->devConns;
  return mscclppSuccess;
}
