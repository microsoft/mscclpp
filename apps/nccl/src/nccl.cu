// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <filesystem>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>
#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif
#include "allgather.hpp"
#include "allreduce.hpp"
#include "broadcast.hpp"
#include "debug.h"
#include "nccl.h"

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

// static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
//                             mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
//                             mscclpp::Transport::IB6, mscclpp::Transport::IB7};

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
  std::vector<mscclpp::SmChannel> smChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
};

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  std::shared_ptr<mscclpp::Executor> executor;
  std::unordered_map<std::string, std::vector<executionPlanInstance>> executionPlans;

  std::unordered_map<channelKey, ChannelInfo> channelInInfos;
  std::unordered_map<channelKey, ChannelInfo> channelOutInfos;
  std::unordered_map<channelKey, ChannelInfo> channelScratchInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;

  uint32_t numScratchBuff;
  uint32_t buffFlag;
};

static size_t ncclTypeSize(ncclDataType_t type) {
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
  // if (rank / nRanksPerNode == peerRank / nRanksPerNode) {
  //   return mscclpp::Transport::CudaIpc;
  // } else {
  //   return IBs[rank % nRanksPerNode];
  // }
  return mscclpp::Transport::CudaIpc;
}

static std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(std::shared_ptr<mscclpp::Communicator> comm, int rank,
                                                                  void* buff, size_t bytes,
                                                                  mscclpp::TransportFlags transport) {
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  mscclpp::RegisteredMemory memory = comm->registerMemory(buff, bytes, transport);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(i, 0));
    comm->sendMemoryOnSetup(memory, i, 0);
  }
  comm->setup();
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(), std::back_inserter(remoteMemories),
                 [](const auto& future) { return future.get(); });
  return remoteMemories;
}

static std::vector<mscclpp::SmChannel> setupSmChannels(ncclComm_t comm,
                                                       const std::vector<mscclpp::RegisteredMemory>& remoteMemories,
                                                       void* src) {
  std::vector<mscclpp::SmChannel> channels;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores = comm->smSemaphores;
  size_t nConnections = comm->connections.size();
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(smSemaphores[idx * nConnections + cid], remoteMemories[cid], src, nullptr);
      }
    }
  }
  return channels;
}

static std::pair<std::string, executionPlanInstance> loadExecutionPlan(const std::string& filename) {
  std::shared_ptr<mscclpp::ExecutionPlan> plan = std::make_shared<mscclpp::ExecutionPlan>(filename);
  std::string collective = plan->collective();
  planKey key{plan->minMessageSize(), plan->maxMessageSize(), plan->isInPlace()};
  return std::make_pair(collective, executionPlanInstance{key, plan});
}

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
                 [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::SmChannel>>(smChannelDeviceHandles.size());
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::SmChannel>>(ptr.get(), smChannelDeviceHandles.data(),
                                                                smChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

static ncclResult_t ncclAllReduceFallback(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                          ncclRedOp_t, ncclComm_t comm, cudaStream_t stream) {
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
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels = nullptr;

  // Creating the channels
  if (count * ncclTypeSize(datatype) <= (1 << 20)) {
    auto sendIt = comm->channelScratchInfos.find(sendKey);
    if (sendIt == comm->channelScratchInfos.end()) {
      std::vector<mscclpp::SmChannel> channels =
          setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>((void*)sendBasePtr));
      ChannelInfo channelInfo{channels, setupSmChannelDeviceHandles(channels)};
      sendIt = comm->channelScratchInfos.emplace(sendKey, channelInfo).first;
    }

    smChannels = sendIt->second.smChannelDeviceHandles.get();
  } else {
    std::vector<mscclpp::RegisteredMemory> remoteMemories;

    auto sendIt = comm->channelInInfos.find(sendKey);
    if (sendIt == comm->channelInInfos.end()) {
      std::vector<mscclpp::SmChannel> channels =
          setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>((void*)sendBasePtr));
      ChannelInfo channelInfo{channels, setupSmChannelDeviceHandles(channels)};
      sendIt = comm->channelInInfos.emplace(sendKey, channelInfo).first;
    }

    auto recvIt = comm->channelOutInfos.find(recvKey);
    if (recvIt == comm->channelOutInfos.end()) {
      remoteMemories =
          setupRemoteMemories(comm->comm, rank, (void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
      std::vector<mscclpp::SmChannel> outChannels =
          setupSmChannels(comm, remoteMemories, const_cast<void*>((void*)recvBasePtr));
      ChannelInfo channelInfo{outChannels, setupSmChannelDeviceHandles(outChannels)};
      recvIt = comm->channelOutInfos.emplace(recvKey, channelInfo).first;
    }

    smChannels = sendIt->second.smChannelDeviceHandles.get();
    smOutChannels = recvIt->second.smChannelDeviceHandles.get();
  }

  switch (datatype) {
    case ncclFloat16:
      CUDACHECK(allreduce((half*)sendbuff, (half*)comm->scratchBuff.get(), (half*)recvbuff, smChannels, smOutChannels,
                          offsetIn, offsetOut, offsetScratch, rank, NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case ncclFloat32:
      CUDACHECK(allreduce((float*)sendbuff, (float*)comm->scratchBuff.get(), (float*)recvbuff, smChannels,
                          smOutChannels, offsetIn, offsetOut, offsetScratch, comm->comm->bootstrap()->getRank(),
                          NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case ncclBfloat16:
      CUDACHECK(allreduce((__bfloat16*)sendbuff, (__bfloat16*)comm->scratchBuff.get(), (__bfloat16*)recvbuff,
                          smChannels, smOutChannels, offsetIn, offsetOut, offsetScratch, rank, NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case ncclInt32:
    case ncclUint32:
      CUDACHECK(allreduce((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, smOutChannels,
                          offsetIn, offsetOut, offsetScratch, comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ncclAllGatherFallback(const void* sendbuff, void* recvbuff, size_t sendcount,
                                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // FallBack for single node
  if (comm->comm->bootstrap()->getNranks() != comm->comm->bootstrap()->getNranksPerNode()) {
    WARN("ncclAllGatherFallback is currently unavailable for multi-node");
    return ncclInvalidUsage;
  }

  // Checking if the parameters are valids
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  // Declarating variables
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)recvbuff));
  size_t offsetOut = (char*)recvbuff - (char*)recvBasePtr;
  channelKey recvKey{(void*)recvBasePtr, recvBytes};
  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;

  auto it = comm->channelOutInfos.find(recvKey);
  if (it == comm->channelOutInfos.end()) {
    std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(
        comm->comm, rank, const_cast<void*>((void*)recvBasePtr), recvBytes, mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(comm, remoteMemories, const_cast<void*>((void*)recvBasePtr));
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    ChannelInfo channelInfo{channels, setupSmChannelDeviceHandles(channels)};
    it = comm->channelOutInfos.emplace(recvKey, channelInfo).first;
  }

  smChannels = it->second.smChannelDeviceHandles.get();
  if ((char*)sendbuff == (char*)recvbuff + rank * sendcount) {
    CUDACHECK(allgather<false>((int*)sendbuff, (int*)nullptr, (int*)recvbuff, smChannels, offsetOut, rank,
                               NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  } else {
    CUDACHECK(allgather<true>((int*)sendbuff, (int*)nullptr, (int*)recvbuff, smChannels, offsetOut, rank,
                              NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  }

  return ncclSuccess;
}

static void ncclCommInitRankFallbackSingleNode(ncclComm* commPtr, std::shared_ptr<mscclpp::Communicator> mscclppComm,
                                               int rank) {
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

  for (int i = 0; i < mscclppComm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    mscclpp::Transport transport = getTransport(rank, i);
    connectionFutures.push_back(mscclppComm->connectOnSetup(i, 0, transport));
  }
  mscclppComm->setup();

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });

  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        smSemaphores.emplace_back(
            std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
      }
    }
  }

  mscclppComm->setup();
  commPtr->connections = std::move(connections);
  commPtr->smSemaphores = std::move(smSemaphores);
  commPtr->buffFlag = 0;
  commPtr->numScratchBuff = 2;
  commPtr->scratchBuff = mscclpp::GpuBuffer(SCRATCH_SIZE).memory();
  commPtr->remoteScratchRegMemories =
      setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);
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
  if (MSCCLPP_UNIQUE_ID_BYTES != NCCL_UNIQUE_ID_BYTES) return ncclInternalError;
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  memcpy(uniqueId, &id, sizeof(ncclUniqueId));
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
                                             ncclConfig_t*) {
  // TODO: implement config
  return ncclCommInitRank(comm, nranks, commId, rank);
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

  // FallBack for single node
  if (mscclppComm->bootstrap()->getNranks() == mscclppComm->bootstrap()->getNranksPerNode())
    ncclCommInitRankFallbackSingleNode(commPtr, mscclppComm, rank);

  const std::string& collectiveDir = mscclpp::env()->executionPlanDir;
  if (collectiveDir != "") {
    if (!std::filesystem::is_directory(collectiveDir)) {
      WARN("The value of the environment variable %s is not a directory", collectiveDir.c_str());
      return ncclInvalidArgument;
    }
    for (const auto& entry : std::filesystem::directory_iterator(collectiveDir)) {
      if (entry.is_regular_file()) {
        auto plan = loadExecutionPlan(entry.path());
        commPtr->executionPlans[plan.first].push_back(plan.second);
      }
    }
  }

  *comm = commPtr;
#if defined(ENABLE_NPKIT)
  if (mscclpp::env()->npkitDumpDir != "") {
    NpKit::Init(rank);
  }
#endif
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
  delete comm;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommAbort(ncclComm_t) {
  // TODO: implement this function
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) {
  // TODO: implement this function
  WARN("ncclCommSplit is currently unavailable");
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

NCCL_API ncclResult_t ncclBroadcastFallback(const void* sendbuff, void* recvbuff, size_t sendcount,
                                            ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) {
    WARN(
        "One or more of the following conditions is met: sendbuff or recvbuff pointer is nullptr, bytes is 0, "
        "or comm is nullptr.");
    return ncclInvalidArgument;
  }

  // Declarating variables
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)recvbuff));
  // size_t offsetOut = (char*)recvbuff - (char*)recvBasePtr;
  size_t offsetOut = 0;
  // channelKey recvKey{(void*)recvBasePtr, recvBytes};
  channelKey recvKey{(void*)0x0, 0};  // Just create the channel once.
  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;

  auto it = comm->channelOutInfos.find(recvKey);
  if (it == comm->channelOutInfos.end()) {
    // std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(
    //     comm->comm, rank, const_cast<void*>((void*)recvBasePtr), recvBytes, mscclpp::Transport::CudaIpc);
    // std::vector<mscclpp::SmChannel> channels =
    //     setupSmChannels(comm, remoteMemories, const_cast<void*>((void*)recvBasePtr));
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>((void*)recvBasePtr));
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    ChannelInfo channelInfo{channels, setupSmChannelDeviceHandles(channels)};
    it = comm->channelOutInfos.emplace(recvKey, channelInfo).first;
  }

  smChannels = it->second.smChannelDeviceHandles.get();
  if ((char*)sendbuff == (char*)recvbuff) {
    CUDACHECK(broadcast<false>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, offsetOut,
                               rank, NRANKS_PER_NODE, root, nRank, bytes / sizeof(int), stream));
  } else {
    CUDACHECK(broadcast<true>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, offsetOut,
                              rank, NRANKS_PER_NODE, root, nRank, bytes / sizeof(int), stream));
  }

  return ncclSuccess;
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

  int rank = comm->comm->bootstrap()->getRank();

  std::vector<executionPlanInstance>& plans = comm->executionPlans["broadcast"];
  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  void* basePtr = (char*)sendbuff;
  bool inPlace = basePtr == recvbuff;
  const size_t totalBytes = bytes;
  for (const auto& p : plans) {
    if (totalBytes >= p.key.minMessageSize && totalBytes < p.key.maxMessageSize && inPlace == p.key.isInPlace) {
      plan = p.plan;
      break;
    }
  }

  if (plan == nullptr) return ncclBroadcastFallback(sendbuff, recvbuff, count, datatype, root, comm, stream);

  switch (datatype) {
    case ncclFloat16:
      comm->executor->execute(rank, (half*)sendbuff, (half*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT16, *plan,
                              stream);
      break;
    case ncclFloat32:
      comm->executor->execute(rank, (float*)sendbuff, (float*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT32, *plan,
                              stream);
      break;
    case ncclBfloat16:
      comm->executor->execute(rank, (__bfloat16*)sendbuff, (__bfloat16*)recvbuff, bytes, bytes,
                              mscclpp::DataType::BFLOAT16, *plan, stream);
      break;
    case ncclInt32:
    case ncclUint32:
      comm->executor->execute(rank, (int*)sendbuff, (int*)recvbuff, bytes, bytes, mscclpp::DataType::UINT32, *plan,
                              stream);
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }

  return ncclSuccess;
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

  if (plan == nullptr)
    return ncclAllReduceFallback(sendbuff, recvbuff, count, datatype, reductionOperation, comm, stream);

  switch (datatype) {
    case ncclFloat16:
      comm->executor->execute(rank, (half*)sendbuff, (half*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT16, *plan,
                              stream, mscclpp::PacketType::LL8);
      break;
    case ncclFloat32:
      comm->executor->execute(rank, (float*)sendbuff, (float*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT32, *plan,
                              stream, mscclpp::PacketType::LL8);
      break;
    case ncclBfloat16:
      comm->executor->execute(rank, (__bfloat16*)sendbuff, (__bfloat16*)recvbuff, bytes, bytes,
                              mscclpp::DataType::BFLOAT16, *plan, stream, mscclpp::PacketType::LL8);
      break;
    case ncclInt32:
    case ncclUint32:
      comm->executor->execute(rank, (int*)sendbuff, (int*)recvbuff, bytes, bytes, mscclpp::DataType::UINT32, *plan,
                              stream, mscclpp::PacketType::LL8);
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                        cudaStream_t) {
  // TODO: implement this function
  WARN("ncclReduceScatter is currently unavailable");
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
  if (plan == nullptr) return ncclAllGatherFallback(sendbuff, recvbuff, sendcount, datatype, comm, stream);

  switch (datatype) {
    case ncclFloat16:
      comm->executor->execute(rank, (half*)sendbuff, (half*)recvbuff, bytes, bytes * nRank, mscclpp::DataType::FLOAT16,
                              *plan, stream);
      break;
    case ncclFloat32:
      comm->executor->execute(rank, (float*)sendbuff, (float*)recvbuff, bytes, bytes * nRank,
                              mscclpp::DataType::FLOAT32, *plan, stream);
      break;
    case ncclBfloat16:
      comm->executor->execute(rank, (__bfloat16*)sendbuff, (__bfloat16*)recvbuff, bytes, bytes * nRank,
                              mscclpp::DataType::BFLOAT16, *plan, stream);
      break;
    case ncclInt32:
    case ncclUint32:
      comm->executor->execute(rank, (int*)sendbuff, (int*)recvbuff, bytes, bytes * nRank, mscclpp::DataType::UINT32,
                              *plan, stream);
      break;
    default:
      WARN("datatype is invalid");
      return ncclInvalidArgument;
  }

  return ncclSuccess;
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
    INFO(MSCCLPP_ALLOC, "Cuda error: %s", e.what());
    return ncclUnhandledCudaError;
  } catch (const mscclpp::CuError& e) {
    INFO(MSCCLPP_ALLOC, "Cu error: %s", e.what());
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
