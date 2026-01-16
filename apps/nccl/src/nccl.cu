// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <filesystem>
#include <functional>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/utils.hpp>
#include <unordered_map>
#include <vector>
#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif
#include <dlfcn.h>
#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>

#include "allgather.hpp"
#include "allreduce.hpp"
#include "broadcast.hpp"
#include "datatype_conversion.hpp"
#include "debug.h"

#define NCCL_API extern "C" __attribute__((visibility("default")))

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

typedef enum mscclppNcclDlopenErr {
  dlopenSuccess = 0,
  dlopenError = 1,
} mscclppNcclDlopenErr_t;

typedef struct _mscclppNcclOps_t {
  ncclResult_t (*CommInitRank)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
  ncclResult_t (*GetUniqueId)(ncclUniqueId* uniqueId);
  ncclResult_t (*CommDestroy)(ncclComm_t comm);
  ncclResult_t (*CommUserRank)(const ncclComm_t, int* rank);
  ncclResult_t (*AllReduce)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*AllGather)(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*Broadcast)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                            ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*ReduceScatter)(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*Reduce)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                         int root, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*Send)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                       cudaStream_t stream);
  ncclResult_t (*Recv)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                       cudaStream_t stream);
  ncclResult_t (*GroupStart)();
  ncclResult_t (*GroupEnd)();
} mscclppNcclOps_t;

mscclppNcclOps_t mscclppNcclOps;
void* mscclppNcclDlHandle = NULL;
bool mscclppNcclDlopenSharedLib = false;

#define QUOTE(symbol) #symbol

#define NCCL_DLSYM(_struct_, _handle_, _prefix_, _function_, _type_)                               \
  do {                                                                                             \
    _struct_._function_ = (_type_)dlsym((_handle_), QUOTE(_prefix_##_function_));                  \
    if (_struct_._function_ == NULL) {                                                             \
      printf("Failed: dlsym error: Cannot open %s: %s\n", QUOTE(_prefix_##_function_), dlerror()); \
      exit(dlopenError);                                                                           \
    }                                                                                              \
  } while (0)

static inline int mscclppNcclDlopenInit() {
  const char* ncclLibPath = mscclpp::env()->ncclSharedLibPath.c_str();
  if (ncclLibPath != nullptr && ncclLibPath[0] != '\0') {
    if (std::filesystem::is_directory(ncclLibPath)) {
      WARN("The value of the environment variable %s is a directory", ncclLibPath);
      return dlopenError;
    }

    mscclppNcclDlHandle = dlopen(ncclLibPath, RTLD_LAZY | RTLD_NODELETE);
    if (!mscclppNcclDlHandle) {
      WARN("Cannot open the shared library specified by MSCCLPP_NCCL_LIB_PATH: %s\n", dlerror());
      return dlopenError;
    }
  } else {
    WARN("The value of MSCCLPP_NCCL_LIB_PATH is empty!\n");
    return dlopenError;
  }

  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommInitRank,
             ncclResult_t(*)(ncclComm_t*, int, ncclUniqueId, int));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GetUniqueId, ncclResult_t(*)(ncclUniqueId*));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommDestroy, ncclResult_t(*)(ncclComm_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, CommUserRank, ncclResult_t(*)(ncclComm_t, int*));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, AllReduce,
             ncclResult_t(*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, AllGather,
             ncclResult_t(*)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Broadcast,
             ncclResult_t(*)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, ReduceScatter,
             ncclResult_t(*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Reduce,
             ncclResult_t(*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Send,
             ncclResult_t(*)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, Recv,
             ncclResult_t(*)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t));
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GroupStart, ncclResult_t(*)());
  NCCL_DLSYM(mscclppNcclOps, mscclppNcclDlHandle, nccl, GroupEnd, ncclResult_t(*)());

  return dlopenSuccess;
}

// No need to call this function, handle will be closed at program exit
[[maybe_unused]] static inline void mscclppNcclDlopenFinalize() {
  if (mscclppNcclDlHandle) {
    dlclose(mscclppNcclDlHandle);
  }
}

static inline int mscclppNcclInFallbackList(const char* collOps, const char* fallbackList) {
  if (strcmp(fallbackList, "all") == 0) {
    return 1;
  }

  char* fallbackListCopy = strdup(fallbackList);
  char* token = strtok(fallbackListCopy, ",");
  while (token != NULL) {
    if (strcmp(collOps, token) == 0) {
      free(fallbackListCopy);
      return 1;
    }
    token = strtok(NULL, ",");
  }

  free(fallbackListCopy);
  return 0;
}

static bool tryLoadNcclSharedLib() {
  if (mscclppNcclDlopenSharedLib) return true;
  if (!mscclpp::env()->ncclSharedLibPath.empty()) {
    if (mscclppNcclDlopenInit() == dlopenSuccess) {
      mscclppNcclDlopenSharedLib = true;
      return true;
    }
  }
  return false;
}

// Declare the global map to store associations between raw pointer and shared pointer
static std::unordered_map<void*, std::shared_ptr<char>> ptrMap;

struct splitCommInfo {
  int color;
  int key;
  int originalRank;
};

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::shared_ptr<mscclpp::Executor> executor;
  std::shared_ptr<mscclpp::AlgorithmCollection> algorithmCollection;
  std::shared_ptr<char> scratchBuffer_;
  const size_t scratchBufferSize_ = (1 << 27);  // 128MB
  std::shared_ptr<mscclpp::ExecutionPlanRegistry> planRegistry_;
  int nRanksPerNode;
  int worldSize;

  void* mscclppNcclComm;
};

static ncclResult_t executeWithPlan(std::shared_ptr<mscclpp::Executor> executor, int rank, ncclDataType_t datatype,
                                    const void* sendbuff, void* recvbuff, size_t sendBytes, size_t recvBytes,
                                    std::shared_ptr<mscclpp::ExecutionPlan> plan, cudaStream_t stream) {
  switch (datatype) {
    case ncclFloat16:
      executor->execute(rank, (half*)sendbuff, (half*)recvbuff, sendBytes, recvBytes, mscclpp::DataType::FLOAT16, *plan,
                        stream);
      break;
    case ncclFloat32:
      executor->execute(rank, (float*)sendbuff, (float*)recvbuff, sendBytes, recvBytes, mscclpp::DataType::FLOAT32,
                        *plan, stream);
      break;
    case ncclBfloat16:
      executor->execute(rank, (__bfloat16*)sendbuff, (__bfloat16*)recvbuff, sendBytes, recvBytes,
                        mscclpp::DataType::BFLOAT16, *plan, stream);
      break;
#if defined(__FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
      executor->execute(rank, (__fp8_e4m3*)sendbuff, (__fp8_e4m3*)recvbuff, sendBytes, recvBytes,
                        mscclpp::DataType::FP8_E4M3, *plan, stream);
      break;
    case ncclFloat8e5m2:
      executor->execute(rank, (__fp8_e5m2*)sendbuff, (__fp8_e5m2*)recvbuff, sendBytes, recvBytes,
                        mscclpp::DataType::FP8_E5M2, *plan, stream);
      break;
#endif
    case ncclInt32:
    case ncclUint32:
      executor->execute(rank, (int*)sendbuff, (int*)recvbuff, sendBytes, recvBytes, mscclpp::DataType::UINT32, *plan,
                        stream);
      break;
    case ncclUint8:
      executor->execute(rank, (uint8_t*)sendbuff, (uint8_t*)recvbuff, sendBytes, recvBytes, mscclpp::DataType::UINT8,
                        *plan, stream);
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGetVersion(int* version) {
  if (version == nullptr) {
    WARN("version is nullptr");
    return ncclInvalidArgument;
  }
  *version = MSCCLPP_VERSION;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) {
    WARN("uniqueId is nullptr");
    return ncclInvalidArgument;
  }
  if (mscclpp::UniqueIdBytes != NCCL_UNIQUE_ID_BYTES) return ncclInternalError;
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  memcpy(uniqueId, &id, sizeof(ncclUniqueId));
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
                                             ncclConfig_t*) {
  // TODO: implement config
  return ncclCommInitRank(comm, nranks, commId, rank);
}

static void registerCustomizedAlgo() {
  auto collectionBuilder = mscclpp::AlgorithmCollectionBuilder::getInstance();
  std::shared_ptr<BroadcastAlgo6> broadcastAlgo6 = std::make_shared<BroadcastAlgo6>();
  collectionBuilder->addAlgorithmBuilder(broadcastAlgo6);

  std::shared_ptr<AllgatherAlgo6> allgatherAlgo6 = std::make_shared<AllgatherAlgo6>();
  std::shared_ptr<AllgatherAlgo8> allgatherAlgo8 = std::make_shared<AllgatherAlgo8>();
  collectionBuilder->addAlgorithmBuilder(allgatherAlgo6);
  // TODO(binyli): remove allgather8 algo, use nccl by default
  collectionBuilder->addAlgorithmBuilder(allgatherAlgo8);

  std::shared_ptr<AllreducePacket> allreduceAllpairAlgo = std::make_shared<AllreducePacket>();
  std::shared_ptr<AllreduceNvls> allreduceNvlsAlgo = std::make_shared<AllreduceNvls>();
  std::shared_ptr<AllreduceNvlsWithCopy> allreduceNvlsWithCopyAlgo = std::make_shared<AllreduceNvlsWithCopy>();
  std::shared_ptr<Allreduce8> allreduceAllreduce8Algo = std::make_shared<Allreduce8>();
  std::shared_ptr<AllreduceNvlsPacket> allreduceNvlsPacketAlgo = std::make_shared<AllreduceNvlsPacket>();
  collectionBuilder->addAlgorithmBuilder(allreduceAllpairAlgo);
  collectionBuilder->addAlgorithmBuilder(allreduceNvlsAlgo);
  collectionBuilder->addAlgorithmBuilder(allreduceNvlsWithCopyAlgo);
  collectionBuilder->addAlgorithmBuilder(allreduceAllreduce8Algo);
  collectionBuilder->addAlgorithmBuilder(allreduceNvlsPacketAlgo);
}

static std::pair<int, int> getDeviceComputeCapability() {
  int device;
  CUDACHECK(cudaGetDevice(&device));
  int major = 0, minor = 0;
  CUDACHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDACHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return std::make_pair(major, minor);
}

static mscclpp::Algorithm algoSelector(
    const std::unordered_map<std::string, std::unordered_map<std::string, mscclpp::Algorithm>>& algoMapByCollective,
    std::string collective, const void* input, void* output, size_t messageSize, mscclpp::DataType dtype,
    int nRanksPerNode, int worldSize) {
  if (nRanksPerNode != worldSize) {
    // Fallback to nccl/rccl when multi-node
    return mscclpp::Algorithm();
  }
  static const bool mscclppDisableChannelCache = mscclpp::env()->disableChannelCache;
  static const bool isNvlsSupported = mscclpp::isNvlsSupported();
  static const std::pair<int, int> deviceComputeCapability = getDeviceComputeCapability();
  bool isCuMemMapAllocated =
      mscclpp::isCuMemMapAllocated(const_cast<void*>(input)) && mscclpp::isCuMemMapAllocated(output);
  bool useNvlsWithZeroCopy = isNvlsSupported && !mscclppDisableChannelCache && isCuMemMapAllocated;
  if (collective == "allgather") {
    if (messageSize <= 32 * (1 << 20)) {
      return algoMapByCollective.at(collective).at("default_allgather6");
    } else {
#if defined(__HIP_PLATFORM_AMD__)
      return algoMapByCollective.at(collective).at("default_allgather6");
#else
      if (!mscclppNcclDlopenSharedLib) {
        return algoMapByCollective.at(collective).at("default_allgather8");
      }
#endif
    }
  }
  if (collective == "allreduce") {
    bool useNvls = isNvlsSupported;
    bool isFp8 = dtype == mscclpp::DataType::FP8_E4M3 || dtype == mscclpp::DataType::FP8_E5M2;
#if !defined(__HIP_PLATFORM_AMD__)
    if (isFp8 && deviceComputeCapability.first < 10) {
      // NVLS does not support FP8 on devices with compute capability < 10
      useNvls = false;
    }
#endif
    if (messageSize <= (1 << 15) && useNvls) {
      return algoMapByCollective.at(collective).at("default_allreduce_nvls_packet");
    }
    if (messageSize <= (1 << 16) || (messageSize <= (1 << 20) && !useNvlsWithZeroCopy)) {
      return algoMapByCollective.at(collective).at("default_allreduce_packet");
    }
    if (useNvls && useNvlsWithZeroCopy) {
      return algoMapByCollective.at(collective).at("default_allreduce_nvls");
    }
    if (useNvls) {
      return algoMapByCollective.at(collective).at("default_allreduce_nvls_with_copy");
    }
#if defined(__HIP_PLATFORM_AMD__)
    return algoMapByCollective.at(collective).at("default_allreduce_allreduce8");
#else
    if (!mscclppNcclDlopenSharedLib) {
      return algoMapByCollective.at(collective).at("default_allreduce_allreduce8");
    }
#endif
  }
  INFO(MSCCLPP_NCCL, "Failed to get algo from customized kernel, fallback to nccl/rccl");
  return mscclpp::Algorithm();
}

std::shared_ptr<mscclpp::ExecutionPlanHandle> executionPlanDefaultSelector(
    const std::vector<std::shared_ptr<mscclpp::ExecutionPlanHandle>> plans, const mscclpp::ExecutionRequest&) {
  if (plans.empty()) {
    INFO(MSCCLPP_NCCL, "No execution plans available for selection");
    return nullptr;
  }
  for (auto plan : plans) {
    if (plan->tags.find("default") == plan->tags.end()) {
      return plan;
    }
  }
  return plans[0];
}

NCCL_API ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  INFO(MSCCLPP_NCCL, "Initializing NCCL communicator for rank %d, world_size=%d", rank, nranks);
  if (comm == nullptr) {
    WARN("comm is nullptr");
    return ncclInvalidArgument;
  }
  if (nranks < 0 || rank < 0 || rank >= nranks) {
    WARN("nranks is %d, rank is %d", nranks, rank);
    return ncclInvalidArgument;
  }
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId id;
  memcpy(id.data(), &commId, sizeof(ncclUniqueId));
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> mscclppComm = std::make_shared<mscclpp::Communicator>(bootstrap);
  ncclComm* commPtr = new ncclComm();

  commPtr->comm = mscclppComm;
  commPtr->scratchBuffer_ = mscclpp::GpuBuffer<char>(commPtr->scratchBufferSize_).memory();
  commPtr->executor = std::make_shared<mscclpp::Executor>(mscclppComm, commPtr->scratchBuffer_);
  commPtr->planRegistry_ = mscclpp::ExecutionPlanRegistry::getInstance();

  commPtr->nRanksPerNode = mscclppComm->bootstrap()->getNranksPerNode();
  commPtr->worldSize = mscclppComm->bootstrap()->getNranks();
  commPtr->planRegistry_->loadDefaultPlans(rank);
  commPtr->planRegistry_->setDefaultSelector(executionPlanDefaultSelector);
  mscclpp::AlgorithmCollectionBuilder::getInstance()->setFallbackAlgorithmSelector(algoSelector);
  registerCustomizedAlgo();
  commPtr->algorithmCollection = mscclpp::AlgorithmCollectionBuilder::getInstance()->build();

  *comm = commPtr;
#if defined(ENABLE_NPKIT)
  if (mscclpp::env()->npkitDumpDir != "") {
    NpKit::Init(rank);
  }
#endif

  const std::string ncclLibPath = mscclpp::env()->ncclSharedLibPath;
  if (!ncclLibPath.empty() && !mscclppNcclDlopenSharedLib) {
    if (!tryLoadNcclSharedLib()) {
      WARN("Failed to load the shared library for nccl/rccl");
      return ncclInternalError;
    }
  }

  if (mscclppNcclDlopenSharedLib == true) {
    ncclUniqueId mscclppNcclUniqueId;
    if (rank == 0) {
      mscclppNcclOps.GetUniqueId(&mscclppNcclUniqueId);
    }
    // After broadcast, mscclppNcclUniqueId on each rank has the same ncclUniqueId
    bootstrap->broadcast(&mscclppNcclUniqueId, sizeof(ncclUniqueId), 0);

    commPtr->mscclppNcclComm = new ncclComm_t();
    if (commPtr->mscclppNcclComm == nullptr) {
      WARN("Failed to allocate memory for mscclppNcclComm");
      return ncclInternalError;
    }
    mscclppNcclOps.CommInitRank(reinterpret_cast<ncclComm_t*>(commPtr->mscclppNcclComm), nranks, mscclppNcclUniqueId,
                                rank);
  }

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int*) {
  if (ndev == 1) {
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);
    return ncclCommInitRank(comm, ndev, Id, 0);
  }
  // TODO: implement this function
  WARN("ncclCommInitAll is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  comm->comm->bootstrap()->barrier();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) {
    WARN("comm is nullptr");
    return ncclInvalidArgument;
  }
#if defined(ENABLE_NPKIT)
  const std::string& npkitDumpDir = mscclpp::env()->npkitDumpDir;
  if (npkitDumpDir != "") {
    NpKit::Dump(npkitDumpDir);
    NpKit::Shutdown();
  }
#endif

  ncclComm_t* mscclppNcclCommPtr = reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm);
  delete comm;
  if (mscclppNcclCommPtr != nullptr) {
    mscclppNcclOps.CommDestroy(*reinterpret_cast<ncclComm_t*>(mscclppNcclCommPtr));
    delete static_cast<ncclComm_t*>(mscclppNcclCommPtr);
  }
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommAbort(ncclComm_t) {
  // TODO: implement this function
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t*) {
  *newcomm = NCCL_COMM_NULL;
  int nRanks = comm->comm->bootstrap()->getNranks();
  int rank = comm->comm->bootstrap()->getRank();
  splitCommInfo info{color, key, comm->comm->bootstrap()->getRank()};
  std::vector<splitCommInfo> infos(nRanks);
  infos[rank] = info;
  comm->comm->bootstrap()->allGather(infos.data(), sizeof(splitCommInfo));
  comm->comm->bootstrap()->barrier();
  std::vector<splitCommInfo> group;
  std::copy_if(infos.begin(), infos.end(), std::back_inserter(group),
               [color](const splitCommInfo& info) { return info.color == color; });
  std::sort(group.begin(), group.end(), [](const splitCommInfo& a, const splitCommInfo& b) { return a.key < b.key; });
  int newRank = std::distance(group.begin(),
                              std::find_if(group.begin(), group.end(),
                                           [rank](const splitCommInfo& info) { return info.originalRank == rank; }));
  int groupSize = group.size();
  ncclUniqueId uniqueId;
  if (newRank == 0) {
    ncclGetUniqueId(&uniqueId);
  }
  std::vector<ncclUniqueId> uniqueIds(nRanks);
  uniqueIds[rank] = uniqueId;
  comm->comm->bootstrap()->allGather(uniqueIds.data(), sizeof(ncclUniqueId));
  comm->comm->bootstrap()->barrier();
  uniqueId = uniqueIds[group.front().originalRank];
  if (color == NCCL_SPLIT_NOCOLOR) {
    return ncclSuccess;
  }
  return ncclCommInitRankConfig(newcomm, groupSize, uniqueId, newRank, nullptr);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t*, int, int, int, ncclUniqueId*, ncclConfig_t*) {
  WARN("ncclCommInitRankScalable is currently unavailable");
  return ncclInternalError;
}

NCCL_API const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with MSCCLPP_DEBUG=INFO for details)";
    case ncclSystemError:
      return "unhandled system error (run with MSCCLPP_DEBUG=INFO for details)";
    case ncclInternalError:
      return "internal error (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclInvalidArgument:
      return "invalid argument (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with MSCCLPP_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

NCCL_API const char* ncclGetLastError(ncclComm_t) {
  // TODO: implement this function
  return "";
}

NCCL_API ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t* asyncError) {
  if (asyncError == nullptr) {
    WARN("asyncError is nullptr");
    return ncclInvalidArgument;
  }
  *asyncError = ncclSuccess;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) {
    WARN("comm is nullptr or count is nullptr");
    return ncclInvalidArgument;
  }
  *count = comm->comm->bootstrap()->getNranks();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) {
    WARN("comm is nullptr or device is nullptr");
    return ncclInvalidArgument;
  }
  *device = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) {
    WARN("comm is nullptr or rank is nullptr");
    return ncclInvalidArgument;
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.CommUserRank(*reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), rank);
  }

  *rank = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t, ncclWindow_t*, int) {
  WARN("ncclCommWindowRegister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) {
  WARN("ncclCommWindowDeregister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*, ncclDataType_t, ncclScalarResidence_t, ncclComm_t) {
  // TODO: implement this function
  WARN("ncclRedOpCreatePreMulSum is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) {
  // TODO: implement this function
  WARN("ncclRedOpDestroy is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Reduce(sendbuff, recvbuff, count, datatype, op, root,
                                 *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }
  WARN("ncclReduce is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                                cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    int root, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  int rank = comm->comm->bootstrap()->getRank();
  if ((sendbuff == nullptr && root == rank) || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  INFO(MSCCLPP_NCCL, "rank %d broadcast sendbuff %p recvbuff %p count %ld, dtype %d, comm: %p", rank, sendbuff,
       recvbuff, count, datatype, comm);

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true && mscclppNcclInFallbackList("broadcast", fallbackList)) {
    return mscclppNcclOps.Broadcast(sendbuff, recvbuff, count, datatype, root,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  static std::unordered_map<std::string, std::vector<uint64_t>> hints{{"root", {static_cast<uint64_t>(root)}}};
  hints["root"][0] = static_cast<uint64_t>(root);
  auto planHandle = comm->planRegistry_->select("broadcast", comm->comm->bootstrap()->getNranks(),
                                                comm->comm->bootstrap()->getNranksPerNode(),
                                                comm->comm->bootstrap()->getRank(), sendbuff, recvbuff, bytes, hints);
  if (planHandle != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, bytes, planHandle->plan, stream);
  }
  mscclpp::DataType mscclppDataType = ncclDataTypeToMscclpp(datatype);
  auto algo = comm->algorithmCollection->selectAlgorithm(
      "broadcast", sendbuff, recvbuff, count * ncclTypeSize(datatype), mscclppDataType,
      comm->comm->bootstrap()->getNranksPerNode(), comm->comm->bootstrap()->getNranks());
  if (!algo.isEmpty()) {
    std::unordered_map<std::string, std::shared_ptr<void>> extras{
        {"root", std::make_shared<int>(root)},
        {"scratch", comm->scratchBuffer_},
        {"scratch_size", std::make_shared<size_t>(comm->scratchBufferSize_)}};
    return static_cast<ncclResult_t>(
        algo.launch(comm->comm, sendbuff, recvbuff, count, mscclppDataType, stream, extras));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Broadcast(sendbuff, recvbuff, count, datatype, root,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN("No FallBack implementation for broadcast");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t reductionOperation, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, count is 0, "
        "datatype is invalid, or comm is nullptr.");
    return ncclInvalidArgument;
  }
  // Declarating variables
  int rank = comm->comm->bootstrap()->getRank();
  INFO(MSCCLPP_NCCL, "rank %d allreduce sendbuff %p recvbuff %p count %ld, dtype %d comm is %p", rank, sendbuff,
       recvbuff, count, datatype, comm);

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib && mscclppNcclInFallbackList("allreduce", fallbackList)) {
    return mscclppNcclOps.AllReduce(sendbuff, recvbuff, count, datatype, reductionOperation,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  auto planHandler = comm->planRegistry_->select("allreduce", comm->comm->bootstrap()->getNranks(),
                                                 comm->comm->bootstrap()->getNranksPerNode(),
                                                 comm->comm->bootstrap()->getRank(), sendbuff, recvbuff, bytes, {});
  if (planHandler != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, bytes, planHandler->plan, stream);
  }
  mscclpp::DataType mscclppDataType = ncclDataTypeToMscclpp(datatype);
  auto algo = comm->algorithmCollection->selectAlgorithm(
      "allreduce", sendbuff, recvbuff, count * ncclTypeSize(datatype), mscclppDataType,
      comm->comm->bootstrap()->getNranksPerNode(), comm->comm->bootstrap()->getNranks());
  if (!algo.isEmpty()) {
    std::unordered_map<std::string, std::shared_ptr<void>> extras{
        {"op", std::make_shared<int>(reductionOperation)},
        {"scratch", comm->scratchBuffer_},
        {"scratch_size", std::make_shared<size_t>(comm->scratchBufferSize_)}};
    return static_cast<ncclResult_t>(
        algo.launch(comm->comm, sendbuff, recvbuff, count, mscclppDataType, stream, extras));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.AllReduce(sendbuff, recvbuff, count, datatype, reductionOperation,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN("No FallBack implementation for AllReduce");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = recvcount * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }

  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  INFO(MSCCLPP_NCCL, "ReduceScatter recvcount: %ld, datatype: %d, op: %d, messageSize: %ld", recvcount, datatype, op,
       bytes * comm->comm->bootstrap()->getNranks());

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true && mscclppNcclInFallbackList("reducescatter", fallbackList)) {
    return mscclppNcclOps.ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();

  auto planHandle = comm->planRegistry_->select("reducescatter", comm->comm->bootstrap()->getNranks(),
                                                comm->comm->bootstrap()->getNranksPerNode(),
                                                comm->comm->bootstrap()->getRank(), sendbuff, recvbuff, bytes, {});
  if (planHandle != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes * nRank, bytes, planHandle->plan,
                           stream);
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN("No FallBack implementation for ReduceScatter");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  INFO(MSCCLPP_NCCL, "rank %d allgather sendbuff %p recvbuff %p count %ld, dtype %d, comm %p", rank, sendbuff, recvbuff,
       sendcount, datatype, comm);

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  if (mscclppNcclDlopenSharedLib == true && mscclppNcclInFallbackList("allgather", fallbackList)) {
    return mscclppNcclOps.AllGather(sendbuff, recvbuff, sendcount, datatype,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  auto planHandle = comm->planRegistry_->select("allgather", comm->comm->bootstrap()->getNranks(),
                                                comm->comm->bootstrap()->getNranksPerNode(),
                                                comm->comm->bootstrap()->getRank(), sendbuff, recvbuff, bytes, {});
  if (planHandle != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, bytes * nRank, planHandle->plan,
                           stream);
  }
  mscclpp::DataType mscclppDataType = ncclDataTypeToMscclpp(datatype);
  auto algo = comm->algorithmCollection->selectAlgorithm(
      "allgather", sendbuff, recvbuff, nRank * sendcount * ncclTypeSize(datatype), mscclppDataType,
      comm->comm->bootstrap()->getNranksPerNode(), comm->comm->bootstrap()->getNranks());
  if (!algo.isEmpty()) {
    std::unordered_map<std::string, std::shared_ptr<void>> extras = {
        {"scratch", comm->scratchBuffer_}, {"scratch_size", std::make_shared<size_t>(comm->scratchBufferSize_)}};
    return static_cast<ncclResult_t>(
        algo.launch(comm->comm, sendbuff, recvbuff, sendcount, mscclppDataType, stream, extras));
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.AllGather(sendbuff, recvbuff, sendcount, datatype,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  WARN("No FallBack implementation for AllGather");
  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                               cudaStream_t stream) {
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Send(sendbuff, count, datatype, peer, *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm),
                               stream);
  }
  WARN("ncclSend is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                               cudaStream_t stream) {
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Recv(recvbuff, count, datatype, peer, *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm),
                               stream);
  }
  WARN("ncclRecv is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    if (sendbuff != recvbuff) {
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }
  // TODO: implement this function
  WARN("ncclAllToAll is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAllv(const void* sendbuff, [[maybe_unused]] const size_t sendcounts[],
                                    const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
                                    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm,
                                    cudaStream_t stream) {
  size_t bytes = recvcounts[0] * ncclTypeSize(datatype);
  if (comm->worldSize == 1) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync((char*)recvbuff + rdispls[0] * ncclTypeSize(datatype),
                                      (const char*)sendbuff + sdispls[0] * ncclTypeSize(datatype), bytes,
                                      cudaMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }
  WARN("ncclAllToAllv is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclGroupStart() {
  tryLoadNcclSharedLib();
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.GroupStart();
  }
  WARN("ncclGroupStart is currently unavailable, return success");
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupEnd() {
  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.GroupEnd();
  }
  WARN("ncclGroupEnd is currently unavailable, return success");
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t*) {
  // TODO: implement this function
  WARN("ncclGroupSimulateEnd is not implemented");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t, void**) {
  // TODO: Implementation
  WARN("ncclCommRegister is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommDeregister(const ncclComm_t, void*) {
  // TODO: Implementation
  WARN("ncclCommDeregister is currently unavailable");
  return ncclInternalError;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    WARN("ptr is nullptr or size is 0");
    return ncclInvalidArgument;
  }
  std::shared_ptr<char> sharedPtr;
  try {
    sharedPtr = mscclpp::GpuBuffer(size).memory();
    if (sharedPtr == nullptr) {
      WARN("Failed to allocate memory via ncclMemAlloc");
      return ncclSystemError;
    }
  } catch (const mscclpp::Error& e) {
    if (e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage) {
      WARN("Invalid usage: %s", e.what());
      return ncclInvalidUsage;
    } else {
      WARN("Internal error: %s", e.what());
      return ncclInternalError;
    }
  } catch (const mscclpp::CudaError& e) {
    WARN("Cuda error: %s", e.what());
    return ncclUnhandledCudaError;
  } catch (const mscclpp::CuError& e) {
    WARN("Cu error: %s", e.what());
    return ncclUnhandledCudaError;
  } catch (const mscclpp::BaseError& e) {
    WARN("Base error: %s", e.what());
    return ncclInternalError;
  }
  ptrMap[sharedPtr.get()] = sharedPtr;

  // Return the pointer
  *ptr = sharedPtr.get();
  return ncclSuccess;
}

ncclResult_t ncclMemFree(void* ptr) {
  auto ptrIt = ptrMap.find(ptr);
  if (ptrIt != ptrMap.end()) {
    ptrMap.erase(ptrIt);
    return ncclSuccess;
  }

  // Pointer not found
  WARN("Pointer not found");
  return ncclInvalidUsage;
}
