#include "mscclpp.h"
#include "bootstrap.h"
#include "core.h"

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
  MSCCLPPCHECKGOTO(mscclppCalloc(&_comm, 1), res, fail);
  _comm->rank = rank;
  _comm->nRanks = nranks;

  MSCCLPPCHECK(bootstrapNetInit(ip_port_pair));
  mscclppBootstrapHandle handle;
  MSCCLPPCHECK(bootstrapGetUniqueId(&handle, rank == 0, ip_port_pair));
  _comm->magic = handle.magic;

  MSCCLPPCHECKGOTO(mscclppCudaHostCalloc((uint32_t **)&_comm->abortFlag, 1), res, fail);
  MSCCLPPCHECK(bootstrapInit(&handle, _comm));
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