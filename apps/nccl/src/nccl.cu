// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <filesystem>
#include <functional>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/utils.hpp>
#include <queue>
#include <sstream>
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

#define NUM_CHANNELS_PER_CONNECTION 64
static constexpr size_t NVLS_BUFFER_SIZE = (1 << 30);

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
  if (mscclpp::env()->ncclSharedLibPath.empty()) {
#if defined(__HIP_PLATFORM_AMD__)
    ncclLibPath = "librccl.so";  // Default RCCL library name
#else
    ncclLibPath = "libnccl.so";  // Default NCCL library name
#endif
  }
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

  return dlopenSuccess;
}

static inline void mscclppNcclDlopenFinalize() {
  if (mscclppNcclDlHandle) {
    dlclose(mscclppNcclDlHandle);
  }
}

static inline int mscclppNcclInFallbackList(const char* collOps, const char* fallbackList) {
  if (fallbackList == nullptr || fallbackList[0] == '\0' || strcmp(fallbackList, "all") == 0) {
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

// Declare the global map to store associations between raw pointer and shared pointer
static std::unordered_map<void*, std::shared_ptr<char>> ptrMap;

struct channelKey {
  const void* buff;
  size_t bytes;
  bool operator==(const channelKey& other) const { return buff == other.buff && bytes == other.bytes; }
};

struct planKey {
  size_t minMessageSize;
  size_t maxMessageSize;
  bool isInPlace;
};

struct executionPlanInstance {
  planKey key;
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
};

namespace std {
template <>
struct hash<channelKey> {
  std::size_t operator()(const channelKey& k) const {
    return std::hash<const void*>()(k.buff) ^ std::hash<size_t>()(k.bytes);
  }
};
}  // namespace std

struct ChannelInfo {
  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles;
};

struct NvlsChannelInfo {
  std::vector<mscclpp::SwitchChannel> nvlsChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> nvlsChannelDeviceHandles;
};

struct splitCommInfo {
  int color;
  int key;
  int originalRank;
};

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnections;
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnectionsOut;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::shared_ptr<mscclpp::Executor> executor;
  std::unordered_map<std::string, std::vector<executionPlanInstance>> executionPlans;

  std::shared_ptr<mscclpp::AlgorithmFactory> algorithmFactory;

  std::unordered_map<channelKey, ChannelInfo> channelInInfos;
  std::unordered_map<channelKey, ChannelInfo> channelOutInfos;
  std::unordered_map<channelKey, ChannelInfo> channelScratchInfos;
  std::unordered_map<channelKey, NvlsChannelInfo> channelNvlsInfos;
  std::shared_ptr<char> scratchBuff;
  mscclpp::RegisteredMemory registeredScratchMemory;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
  std::vector<ChannelInfo> channelInfos;

  uint32_t numScratchBuff;
  uint32_t buffFlag;

  int nRanksPerNode;

  std::shared_ptr<uint32_t> deviceFlag7;
  std::shared_ptr<uint32_t> deviceFlag28;
  std::shared_ptr<uint32_t> deviceFlag56;

  void* mscclppNcclComm;
};

size_t ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
      return 4;
    case ncclInt64:
    case ncclUint64:
      return 8;
    case ncclFloat32:
      return 4;
    case ncclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      return 2;
#endif  // defined(__CUDA_BF16_TYPES_EXIST__)
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFp8E4M3:
    case ncclFp8E5M2:
      return 1;
#endif  // defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclNumTypes:
      return 0;
  }
  return 0;
}

static mscclpp::Transport getTransport(int, int) {
  return mscclpp::Transport::CudaIpc;
}


static std::vector<mscclpp::MemoryChannel> setupMemoryChannels(
    ncclComm_t comm, const std::vector<mscclpp::RegisteredMemory>& remoteMemories,
    mscclpp::RegisteredMemory localMemory) {
  std::vector<mscclpp::MemoryChannel> channels;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memorySemaphores = comm->memorySemaphores;
  size_t nConnections = comm->connections.size();
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nConnections + cid], remoteMemories[cid], localMemory, nullptr);
      }
    }
  }
  return channels;
}

static std::vector<std::shared_ptr<mscclpp::NvlsConnection>> setupNvlsConnections(ncclComm_t comm, size_t size) {
  // for nvls connection
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnections;
  int nRanks = comm->comm->bootstrap()->getNranks();
  std::vector<int> ranks;
  for (int i = 0; i < nRanks; i++) {
    ranks.push_back(i);
  }
  for (int i = 0; i < NUM_NVLS_CONNECTION; i++) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = mscclpp::connectNvlsCollective(comm->comm, ranks, size);
    nvlsConnections.push_back(nvlsConnection);
  }
  return nvlsConnections;
}

static std::vector<mscclpp::SwitchChannel> setupNvlsChannels(
    std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns, void* buffer, size_t bufferSize) {
  std::vector<mscclpp::SwitchChannel> channels;

  for (size_t idx = 0; idx < NUM_NVLS_CONNECTION; ++idx) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = conns[idx];
    mscclpp::SwitchChannel SwitchChannel = nvlsConnection->bindAllocatedMemory((CUdeviceptr)buffer, bufferSize);
    channels.push_back(SwitchChannel);
  }
  return channels;
}

static std::pair<std::string, executionPlanInstance> loadExecutionPlan(const std::string& filename, int rank) {
  std::shared_ptr<mscclpp::ExecutionPlan> plan = std::make_shared<mscclpp::ExecutionPlan>(filename, rank);
  std::string collective = plan->collective();
  planKey key{plan->minMessageSize(), plan->maxMessageSize(), plan->isInPlace()};
  return std::make_pair(collective, executionPlanInstance{key, plan});
}

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
    case ncclInt32:
    case ncclUint32:
      executor->execute(rank, (int*)sendbuff, (int*)recvbuff, sendBytes, recvBytes, mscclpp::DataType::UINT32, *plan,
                        stream);
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> setupNvlsChannelDeviceHandles(
    const std::vector<mscclpp::SwitchChannel>& nvlsChannels) {
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> ptr =
      mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(nvlsChannels.size());
  std::vector<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> nvlsChannelDeviceHandles;
  std::transform(nvlsChannels.begin(), nvlsChannels.end(), std::back_inserter(nvlsChannelDeviceHandles),
                 [](const mscclpp::SwitchChannel& nvlsChannel) { return mscclpp::deviceHandle(nvlsChannel); });
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(
      ptr.get(), nvlsChannelDeviceHandles.data(), nvlsChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

static ncclResult_t ncclAllReduceFallback(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                          ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  // FallBack for single node
  if (comm->comm->bootstrap()->getNranks() != comm->comm->bootstrap()->getNranksPerNode()) {
    WARN("ncclAllReduceFallback is currently unavailable for multi-node");
    return ncclInvalidUsage;
  }

  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, count is 0, "
        "datatype is invalid, or comm is nullptr.");
    return ncclInvalidArgument;
  }

  // Declarating variables
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)sendbuff));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)recvbuff));
  size_t offsetIn = (char*)sendbuff - (char*)sendBasePtr;
  size_t offsetOut = (char*)recvbuff - (char*)recvBasePtr;
  uint32_t scratchBuffIdx = (++(comm->buffFlag)) % comm->numScratchBuff;
  size_t offsetScratch = (SCRATCH_SIZE / comm->numScratchBuff) * scratchBuffIdx;
  int rank = comm->comm->bootstrap()->getRank();
  channelKey sendKey{(void*)sendBasePtr, sendBytes};
  channelKey recvKey{(void*)recvBasePtr, recvBytes};
  mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryOutChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels = nullptr;
  size_t bytes = count * ncclTypeSize(datatype);
  bool useNvlsWithZeroCopy = mscclpp::isNvlsSupported() && !mscclppDisableChannelCache;
  bool useNvlsWithCopy = mscclpp::isNvlsSupported() && mscclppDisableChannelCache;

  // Creating the channels
  if (useNvlsWithZeroCopy) {
    auto nvlsIt = comm->channelNvlsInfos.find(sendKey);
    if (nvlsIt == comm->channelNvlsInfos.end()) {
      std::vector<mscclpp::SwitchChannel> channels =
          setupNvlsChannels(comm->nvlsConnections, (void*)sendBasePtr, sendBytes);
      NvlsChannelInfo channelInfo{channels, setupNvlsChannelDeviceHandles(channels)};
      nvlsIt = comm->channelNvlsInfos.emplace(sendKey, channelInfo).first;
    }
    nvlsChannels = nvlsIt->second.nvlsChannelDeviceHandles.get();
    if (recvbuff != sendbuff) {
      auto nvlsOutIt = comm->channelNvlsInfos.find(recvKey);
      if (nvlsOutIt == comm->channelNvlsInfos.end()) {
        std::vector<mscclpp::SwitchChannel> channels =
            setupNvlsChannels(comm->nvlsConnectionsOut, (void*)recvBasePtr, recvBytes);
        NvlsChannelInfo channelInfo{channels, setupNvlsChannelDeviceHandles(channels)};
        nvlsOutIt = comm->channelNvlsInfos.emplace(recvKey, channelInfo).first;
      }
      nvlsOutChannels = nvlsOutIt->second.nvlsChannelDeviceHandles.get();
    } else {
      nvlsOutChannels = nvlsChannels;
    }
  }

  if (useNvlsWithCopy) {
    channelKey sendKey{(void*)(comm->scratchBuff.get()), SCRATCH_SIZE};
    auto nvlsIt = comm->channelNvlsInfos.find(sendKey);
    if (nvlsIt == comm->channelNvlsInfos.end()) {
      std::vector<mscclpp::SwitchChannel> channels =
          setupNvlsChannels(comm->nvlsConnections, (void*)comm->scratchBuff.get(), SCRATCH_SIZE);
      NvlsChannelInfo channelInfo{channels, setupNvlsChannelDeviceHandles(channels)};
      nvlsIt = comm->channelNvlsInfos.emplace(sendKey, channelInfo).first;
    }
    nvlsChannels = nvlsIt->second.nvlsChannelDeviceHandles.get();
  }

  if (count * ncclTypeSize(datatype) <= (1 << 20) || mscclpp::isNvlsSupported()) {
    auto sendIt = comm->channelScratchInfos.find(sendKey);
    if (sendIt == comm->channelScratchInfos.end()) {
      mscclpp::RegisteredMemory localMemory =
          comm->comm->registerMemory((void*)sendBasePtr, sendBytes, mscclpp::Transport::CudaIpc);
      std::vector<mscclpp::MemoryChannel> channels =
          setupMemoryChannels(comm, comm->remoteScratchRegMemories, localMemory);
      ChannelInfo channelInfo{channels, setupMemoryChannelDeviceHandles(channels)};
      sendIt = comm->channelScratchInfos.emplace(sendKey, channelInfo).first;
    }

    memoryChannels = sendIt->second.memoryChannelDeviceHandles.get();
  } else {
    std::vector<mscclpp::RegisteredMemory> remoteMemories;

    auto sendIt = comm->channelInInfos.find(sendKey);
    if (sendIt == comm->channelInInfos.end()) {
      mscclpp::RegisteredMemory localMemory =
          comm->comm->registerMemory((void*)sendBasePtr, sendBytes, mscclpp::Transport::CudaIpc);
      std::vector<mscclpp::MemoryChannel> channels =
          setupMemoryChannels(comm, comm->remoteScratchRegMemories, localMemory);
      ChannelInfo channelInfo{channels, setupMemoryChannelDeviceHandles(channels)};
      sendIt = comm->channelInInfos.emplace(sendKey, channelInfo).first;
    }

    auto recvIt = comm->channelOutInfos.find(recvKey);
    if (mscclppDisableChannelCache == true || recvIt == comm->channelOutInfos.end()) {
      if (mscclppDisableChannelCache == true) {
        recvBytes = bytes;
        recvBasePtr = (CUdeviceptr)recvbuff;
        offsetOut = 0;
      }
      mscclpp::RegisteredMemory localMemory =
          comm->comm->registerMemory((void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
      remoteMemories = setupRemoteMemories(comm->comm, rank, localMemory);
      std::vector<mscclpp::MemoryChannel> outChannels = setupMemoryChannels(comm, remoteMemories, localMemory);
      ChannelInfo channelInfo{outChannels, setupMemoryChannelDeviceHandles(outChannels)};
      recvIt = comm->channelOutInfos.emplace(recvKey, channelInfo).first;
      if (mscclppDisableChannelCache == true) {
        comm->channelInfos.push_back(channelInfo);
      }
    }

    memoryChannels = sendIt->second.memoryChannelDeviceHandles.get();
    memoryOutChannels = mscclppDisableChannelCache == true ? comm->channelInfos.back().memoryChannelDeviceHandles.get()
                                                           : recvIt->second.memoryChannelDeviceHandles.get();
  }

  Op reduceOp = getReduceOp(op);
  std::function<cudaError_t(const void*, void*, void*, mscclpp::DeviceHandle<mscclpp::MemoryChannel>*,
                            mscclpp::DeviceHandle<mscclpp::MemoryChannel>*,
                            mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                            mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int, int, int,
                            size_t, cudaStream_t, uint32_t*, uint32_t*, uint32_t*, int)>
      allreduceFunc;
  if (reduceOp == SUM) {
    if (datatype == ncclFloat16) {
      allreduceFunc = allreduce<SUM, half>;
    } else if (datatype == ncclFloat32) {
      allreduceFunc = allreduce<SUM, float>;
    } else if (datatype == ncclBfloat16) {
      allreduceFunc = allreduce<SUM, __bfloat16>;
    } else if (datatype == ncclInt32 || datatype == ncclUint32) {
      allreduceFunc = allreduce<SUM, int>;
    } else {
      WARN("datatype is invalid, datatype: %d", datatype);
      return ncclInvalidArgument;
    }
  } else if (reduceOp == MIN) {
    if (datatype == ncclFloat16) {
      allreduceFunc = allreduce<MIN, half>;
    } else if (datatype == ncclFloat32) {
      allreduceFunc = allreduce<MIN, float>;
    } else if (datatype == ncclBfloat16) {
      allreduceFunc = allreduce<MIN, __bfloat16>;
    } else if (datatype == ncclInt32 || datatype == ncclUint32) {
      allreduceFunc = allreduce<MIN, int>;
    } else {
      WARN("datatype is invalid, datatype: %d", datatype);
      return ncclInvalidArgument;
    }
  }
  CUDACHECK(allreduceFunc(sendbuff, comm->scratchBuff.get(), recvbuff, memoryChannels, memoryOutChannels, nvlsChannels,
                          nvlsOutChannels, offsetIn, offsetOut, offsetScratch, comm->comm->bootstrap()->getRank(),
                          comm->nRanksPerNode, comm->comm->bootstrap()->getNranks(), count, stream,
                          (uint32_t*)comm->deviceFlag7.get(), (uint32_t*)comm->deviceFlag28.get(),
                          (uint32_t*)comm->deviceFlag56.get(), comm->numScratchBuff));
  return ncclSuccess;
}

static void ncclCommInitRankFallbackSingleNode(ncclComm* commPtr, std::shared_ptr<mscclpp::Communicator> mscclppComm,
                                               int rank) {
  if (mscclpp::isNvlsSupported()) {
    commPtr->nvlsConnections = setupNvlsConnections(commPtr, NVLS_BUFFER_SIZE);
    commPtr->nvlsConnectionsOut = setupNvlsConnections(commPtr, NVLS_BUFFER_SIZE);
  }

  std::vector<std::shared_future<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
  for (int i = 0; i < mscclppComm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    mscclpp::Transport transport = getTransport(rank, i);
    connectionFutures.push_back(mscclppComm->connect(transport, i));
  }
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });

  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        memorySemaphores.emplace_back(
            std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
      }
    }
  }

  commPtr->connections = std::move(connections);
  commPtr->memorySemaphores = std::move(memorySemaphores);
  commPtr->buffFlag = 0;
  commPtr->numScratchBuff = 2;
  commPtr->scratchBuff = mscclpp::GpuBuffer<char>(SCRATCH_SIZE).memory();
  commPtr->registeredScratchMemory =
      commPtr->comm->registerMemory(commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);
  commPtr->remoteScratchRegMemories = setupRemoteMemories(commPtr->comm, rank, commPtr->registeredScratchMemory);

  commPtr->deviceFlag7 = mscclpp::detail::gpuCallocShared<uint32_t>(7);
  commPtr->deviceFlag28 = mscclpp::detail::gpuCallocShared<uint32_t>(28);
  commPtr->deviceFlag56 = mscclpp::detail::gpuCallocShared<uint32_t>(56);

  std::vector<uint32_t> initFlag(56);
  for (int i = 0; i < 56; ++i) {
    initFlag[i] = 1;
  }

  mscclpp::gpuMemcpy<uint32_t>(commPtr->deviceFlag7.get(), initFlag.data(), 7, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(commPtr->deviceFlag28.get(), initFlag.data(), 28, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(commPtr->deviceFlag56.get(), initFlag.data(), 56, cudaMemcpyHostToDevice);
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

static void registerCustomizedAlgo(std::shared_ptr<mscclpp::Communicator> comm) {
  std::shared_ptr<BroadcastAlgo6> broadcastAlgo6 = std::make_shared<BroadcastAlgo6>();
  broadcastAlgo6->registerAlgorithm(comm);

  std::shared_ptr<AllgatherAlgo6> allgatherAlgo6 = std::make_shared<AllgatherAlgo6>();
  std::shared_ptr<AllgatherAlgo8> allgatherAlgo8 = std::make_shared<AllgatherAlgo8>();
  allgatherAlgo6->registerAlgorithm(comm);
  allgatherAlgo8->registerAlgorithm(comm);

  std::shared_ptr<AllreducePacket> allreduceAllpairAlgo = std::make_shared<AllreducePacket>();
  std::shared_ptr<AllreduceNvls> allreduceNvlsAlgo = std::make_shared<AllreduceNvls>(comm);
  allreduceAllpairAlgo->registerAlgorithm(comm);
  allreduceNvlsAlgo->registerAlgorithm(comm);
}

static mscclpp::Algorithm algoSelector(
    const std::unordered_map<std::string, std::unordered_map<std::string, mscclpp::Algorithm>>& algoMapByCollective,
    std::string collective, size_t messageSizes, const void* input, void* output) {
  bool mscclppDisableChannelCache = mscclpp::env()->disableChannelCache;
  bool useNvlsWithZeroCopy = mscclpp::isNvlsSupported() && !mscclppDisableChannelCache;
  if (collective == "broadcast") {
    return algoMapByCollective.at(collective).at("default_broadcast6");
  }
  if (collective == "allgather") {
    if (messageSizes <= 32 * (1 << 20)) {
      return algoMapByCollective.at(collective).at("default_allgather6");
    } else {
#if defined(__HIP_PLATFORM_AMD__)
      return algoMapByCollective.at(collective).at("default_allgather6");
#else
      return algoMapByCollective.at(collective).at("default_allgather8");
#endif
    }
  }
  if (collective == "allreduce") {
    // if (messageSizes <= (1 << 16) || (messageSizes <= (1 << 20) && !useNvlsWithZeroCopy)) {
    //   return algoMapByCollective.at(collective).at("default_allreduce_packet");
    // } else if (useNvlsWithZeroCopy) {
      return algoMapByCollective.at(collective).at("default_allreduce_nvls");
    // }
  }
  INFO(MSCCLPP_NCCL, "Failed to get algo from customized kernel, fallback to nccl");
  return mscclpp::Algorithm();
}

NCCL_API ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
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
  commPtr->executor = std::make_shared<mscclpp::Executor>(mscclppComm);
  commPtr->nRanksPerNode = mscclppComm->bootstrap()->getNranksPerNode();

  commPtr->algorithmFactory = mscclpp::AlgorithmFactory::getInstance();
  if (!commPtr->algorithmFactory->hasAlgorithmSelector()) {
    commPtr->algorithmFactory->setAlgorithmSelector(algoSelector);
  }

  const std::string& collectiveDir = mscclpp::env()->executionPlanDir;
  if (collectiveDir != "") {
    if (!std::filesystem::is_directory(collectiveDir)) {
      WARN("The value of the environment variable %s is not a directory", collectiveDir.c_str());
      return ncclInvalidArgument;
    }
    for (const auto& entry : std::filesystem::directory_iterator(collectiveDir)) {
      if (entry.is_regular_file()) {
        auto plan = loadExecutionPlan(entry.path(), rank);
        commPtr->executionPlans[plan.first].push_back(plan.second);
      }
    }
  }

  registerCustomizedAlgo(mscclppComm);

  *comm = commPtr;
#if defined(ENABLE_NPKIT)
  if (mscclpp::env()->npkitDumpDir != "") {
    NpKit::Init(rank);
  }
#endif

  const bool mscclppEnableNcclFallback = mscclpp::env()->enableNcclFallback;
  if (mscclppNcclDlHandle == NULL) {
    int dlopenStatus = mscclppNcclDlopenInit();
    if (dlopenStatus == dlopenSuccess) {
      mscclppNcclDlopenSharedLib = true;
    } else {
      if (mscclppEnableNcclFallback == true) {
        return ncclInternalError;
      }
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
  ncclCommInitRankFallbackSingleNode(commPtr, mscclppComm, rank);

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitAll(ncclComm_t*, int, const int*) {
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

  if (mscclppNcclDlopenSharedLib == true) {
    mscclppNcclOps.CommDestroy(*reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm));
    mscclppNcclDlopenFinalize();
    delete static_cast<ncclComm_t*>(comm->mscclppNcclComm);
  }

  comm->algorithmFactory->destroy();
  delete comm;
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

NCCL_API ncclResult_t ncclReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t,
                                 cudaStream_t) {
  // TODO: implement this function
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
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  const bool mscclppEnableNcclFallback = mscclpp::env()->enableNcclFallback;
  if (mscclppEnableNcclFallback == true && mscclppNcclInFallbackList("broadcast", fallbackList)) {
    return mscclppNcclOps.Broadcast(sendbuff, recvbuff, count, datatype, root,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  int rank = comm->comm->bootstrap()->getRank();

  std::vector<executionPlanInstance>& plans = comm->executionPlans["broadcast"];
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  bool inPlace = sendbuff == recvbuff;
  for (const auto& p : plans) {
    if (bytes >= p.key.minMessageSize && bytes < p.key.maxMessageSize && inPlace == p.key.isInPlace) {
      plan = p.plan;
      break;
    }
  }

  if (plan != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, bytes, plan, stream);
  }
  auto algo =
      comm->algorithmFactory->selectAlgorithm("broadcast", count * ncclTypeSize(datatype), sendbuff, recvbuff);
  if (!algo.isEmpty()) {
    std::unordered_map<std::string, std::shared_ptr<void>> extras;
    extras.insert({"root", std::make_shared<int>(root)});
    return algo.launch(sendbuff, recvbuff, count, datatype, stream, extras);
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.Broadcast(sendbuff, recvbuff, count, datatype, root,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t reductionOperation, ncclComm_t comm, cudaStream_t stream) {
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, count is 0, "
        "datatype is invalid, or comm is nullptr.");
    return ncclInvalidArgument;
  }

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  const bool mscclppEnableNcclFallback = mscclpp::env()->enableNcclFallback;
  if (mscclppEnableNcclFallback && mscclppNcclInFallbackList("allreduce", fallbackList)) {
    return mscclppNcclOps.AllReduce(sendbuff, recvbuff, count, datatype, reductionOperation,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  // Declarating variables
  size_t bytes = count * ncclTypeSize(datatype);
  int rank = comm->comm->bootstrap()->getRank();

  std::vector<executionPlanInstance>& plans = comm->executionPlans["allreduce"];
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  bool inPlace = sendbuff == recvbuff;
  for (const auto& p : plans) {
    if (bytes >= p.key.minMessageSize && bytes < p.key.maxMessageSize && inPlace == p.key.isInPlace) {
      plan = p.plan;
      break;
    }
  }

  if (plan != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, bytes, plan, stream);
  }

  return ncclAllReduceFallback(sendbuff, recvbuff, count, datatype, reductionOperation, comm, stream);

  // auto algo = comm->algorithmFactory->selectAlgorithm("allreduce", count * ncclTypeSize(datatype), sendbuff, recvbuff);
  // if (!algo.isEmpty()) {
  //   std::unordered_map<std::string, std::shared_ptr<void>> extras{{"op", std::make_shared<int>(reductionOperation)}};
  //   return algo.launch(sendbuff, recvbuff, count, datatype, stream, extras);
  // }

  // if (mscclppNcclDlopenSharedLib == true) {
  //   return mscclppNcclOps.AllReduce(sendbuff, recvbuff, count, datatype, reductionOperation,
  //                                   *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  // }

  // return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = recvcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  const bool mscclppEnableNcclFallback = mscclpp::env()->enableNcclFallback;
  if (mscclppEnableNcclFallback == true && mscclppNcclInFallbackList("reducescatter", fallbackList)) {
    return mscclppNcclOps.ReduceScatter(sendbuff, recvbuff, recvcount, datatype, op,
                                        *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();

  std::vector<executionPlanInstance>& plans = comm->executionPlans["reducescatter"];
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  void* basePtr = (char*)sendbuff + rank * bytes;
  bool inPlace = basePtr == recvbuff;
  const size_t totalBytes = bytes * nRank;
  for (const auto& p : plans) {
    if (totalBytes >= p.key.minMessageSize && totalBytes < p.key.maxMessageSize && inPlace == p.key.isInPlace) {
      plan = p.plan;
      break;
    }
  }

  if (plan != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, totalBytes, bytes, plan, stream);
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
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  const char* fallbackList = mscclpp::env()->forceNcclFallbackOperation.c_str();
  const bool mscclppEnableNcclFallback = mscclpp::env()->enableNcclFallback;
  if (mscclppEnableNcclFallback == true && mscclppNcclInFallbackList("allgather", fallbackList)) {
    return mscclppNcclOps.AllGather(sendbuff, recvbuff, sendcount, datatype,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();

  std::vector<executionPlanInstance>& plans = comm->executionPlans["allgather"];
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  void* basePtr = (char*)sendbuff - rank * bytes;
  bool inPlace = basePtr == recvbuff;
  const size_t totalBytes = bytes * nRank;
  for (const auto& p : plans) {
    if (totalBytes >= p.key.minMessageSize && totalBytes < p.key.maxMessageSize && inPlace == p.key.isInPlace) {
      plan = p.plan;
      break;
    }
  }

  if (plan != nullptr) {
    return executeWithPlan(comm->executor, rank, datatype, sendbuff, recvbuff, bytes, totalBytes, plan, stream);
  }

  auto algo = comm->algorithmFactory->selectAlgorithm("allgather", sendcount * ncclTypeSize(datatype), sendbuff, recvbuff);
  if (!algo.isEmpty()) {
    std::unordered_map<std::string, std::shared_ptr<void>> extras;
    return algo.launch(sendbuff, recvbuff, sendcount, datatype, stream, extras);
  }

  if (mscclppNcclDlopenSharedLib == true) {
    return mscclppNcclOps.AllGather(sendbuff, recvbuff, sendcount, datatype,
                                    *reinterpret_cast<ncclComm_t*>(comm->mscclppNcclComm), stream);
  }

  return ncclInvalidUsage;
}

NCCL_API ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  WARN("ncclSend is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  WARN("ncclRecv is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  WARN("ncclAllToAll is currently unavailable");
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclGroupStart() {
  // Do nothing
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGroupEnd() {
  // Do nothing
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t, void**) {
  // TODO: Implementation
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommDeregister(const ncclComm_t, void*) {
  // TODO: Implementation
  return ncclSuccess;
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
      INFO(MSCCLPP_ALLOC, "Failed to allocate memory");
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
