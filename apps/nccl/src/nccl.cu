// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "allgather.hpp"
#include "allreduce.hpp"
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

struct channelKey {
  const void* buff;
  size_t bytes;
  bool operator==(const channelKey& other) const { return buff == other.buff && bytes == other.bytes; }
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
  std::shared_ptr<mscclpp::ExecutionPlan> allReducePacketIPPlan, allReducePacketOPPlan, allReduceIPPlan,
      allReduceOPPlan, allGatherIPPlan, allGatherOPPlan, allGatherPacketIPPlan, allGatherPacketOPPlan;

  std::unordered_map<channelKey, ChannelInfo> channelInInfos;
  std::unordered_map<channelKey, ChannelInfo> channelOutInfos;
  std::unordered_map<channelKey, ChannelInfo> channelScratchInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;

  size_t allReduceSmallMessageSizeBoundary, allReduceLargeMessageSizeBoundary;
  size_t allGatherSmallMessageSizeBoundary, allGatherLargeMessageSizeBoundary;
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

double parseSize(const char* value) {
  std::string valueStr(value);
  std::istringstream iss(valueStr);
  long long int units;
  double size;
  char size_lit = 0;

  if (iss >> size) {
    iss >> std::ws;  // eat whitespace
    iss >> size_lit;
  } else {
    return -1.0;
  }

  if (size_lit != 0 && !std::isspace(size_lit)) {
    switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024 * 1024 * 1024;
        break;
      case 'M':
      case 'm':
        units = 1024 * 1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
    };
  } else {
    units = 1;
  }
  return size * units;
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

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
                 [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(smChannelDeviceHandles.size());
  mscclpp::memcpyCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(ptr.get(), smChannelDeviceHandles.data(),
                                                                 smChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

static ncclResult_t ncclAllReduceFallback(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                          ncclRedOp_t, ncclComm_t comm, cudaStream_t stream) {
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr)
    return ncclInvalidArgument;

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
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ncclAllGatherFallback(const void* sendbuff, void* recvbuff, size_t sendcount,
                                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;

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

NCCL_API ncclResult_t ncclGetVersion(int* version) {
  if (version == nullptr) return ncclInvalidArgument;
  *version = MSCCLPP_VERSION;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) return ncclInvalidArgument;
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
  if (comm == nullptr) return ncclInvalidArgument;
  if (nranks < 0 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId id;
  memcpy(id.data(), &commId, sizeof(ncclUniqueId));
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> mscclppComm = std::make_shared<mscclpp::Communicator>(bootstrap);
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

  ncclComm* commPtr = new ncclComm();
  commPtr->comm = mscclppComm;
  commPtr->connections = std::move(connections);
  commPtr->smSemaphores = std::move(smSemaphores);
  commPtr->buffFlag = 0;
  commPtr->numScratchBuff = 2;
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(SCRATCH_SIZE);
  commPtr->remoteScratchRegMemories =
      setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);
  commPtr->executor = std::make_shared<mscclpp::Executor>(mscclppComm);

  if (getenv("ALLREDUCEPKT_IP_JSON_FILE"))
    commPtr->allReducePacketIPPlan = std::make_shared<mscclpp::ExecutionPlan>(
        mscclpp::ExecutionPlan("allreduce_packet", getenv("ALLREDUCEPKT_IP_JSON_FILE")));
  if (getenv("ALLREDUCEPKT_OP_JSON_FILE"))
    commPtr->allReducePacketOPPlan = std::make_shared<mscclpp::ExecutionPlan>(
        mscclpp::ExecutionPlan("allreduce_packet", getenv("ALLREDUCEPKT_OP_JSON_FILE")));
  if (getenv("ALLREDUCE_IP_JSON_FILE"))
    commPtr->allReduceIPPlan =
        std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allreduce", getenv("ALLREDUCE_IP_JSON_FILE")));
  if (getenv("ALLREDUCE_OP_JSON_FILE"))
    commPtr->allReduceOPPlan =
        std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allreduce", getenv("ALLREDUCE_OP_JSON_FILE")));
  if (getenv("ALLREDUCE_SMALL_MSG_BOUNDARY"))
    commPtr->allReduceSmallMessageSizeBoundary = parseSize(getenv("ALLREDUCE_SMALL_MSG_BOUNDARY"));
  else
    commPtr->allReduceSmallMessageSizeBoundary = 16 * (1 << 10);
  if (getenv("ALLREDUCE_LARGE_MSG_BOUNDARY"))
    commPtr->allReduceLargeMessageSizeBoundary = parseSize(getenv("ALLREDUCE_LARGE_MSG_BOUNDARY"));
  else
    commPtr->allReduceLargeMessageSizeBoundary = 1 << 20;

  if (getenv("ALLGATHERPKT_IP_JSON_FILE"))
    commPtr->allGatherPacketIPPlan = std::make_shared<mscclpp::ExecutionPlan>(
        mscclpp::ExecutionPlan("allgather_pkt", getenv("ALLGATHERPKT_IP_JSON_FILE")));
  if (getenv("ALLGATHERPKT_OP_JSON_FILE"))
    commPtr->allGatherPacketOPPlan = std::make_shared<mscclpp::ExecutionPlan>(
        mscclpp::ExecutionPlan("allgather_pkt", getenv("ALLGATHERPKT_OP_JSON_FILE")));
  if (getenv("ALLGATHER_IP_JSON_FILE"))
    commPtr->allGatherIPPlan =
        std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allgather", getenv("ALLGATHER_IP_JSON_FILE")));
  if (getenv("ALLGATHER_OP_JSON_FILE"))
    commPtr->allGatherOPPlan =
        std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allgather", getenv("ALLGATHER_OP_JSON_FILE")));
  if (getenv("ALLGATHER_SMALL_MSG_BOUNDARY"))
    commPtr->allGatherSmallMessageSizeBoundary = parseSize(getenv("ALLGATHER_SMALL_MSG_BOUNDARY"));
  else
    commPtr->allGatherSmallMessageSizeBoundary = (1 << 10);
  if (getenv("ALLGATHER_LARGE_MSG_BOUNDARY"))
    commPtr->allGatherLargeMessageSizeBoundary = parseSize(getenv("ALLGATHER_LARGE_MSG_BOUNDARY"));
  else
    commPtr->allGatherLargeMessageSizeBoundary = 1 << 20;

  if (commPtr->allReduceSmallMessageSizeBoundary > commPtr->allReduceLargeMessageSizeBoundary)
    return ncclInvalidArgument;
  if (commPtr->allGatherSmallMessageSizeBoundary > commPtr->allGatherLargeMessageSizeBoundary)
    return ncclInvalidArgument;

  *comm = commPtr;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitAll(ncclComm_t*, int, const int*) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  comm->comm->bootstrap()->barrier();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  delete comm;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommAbort(ncclComm_t) {
  // TODO: implement this function
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
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
  if (asyncError == nullptr) return ncclInvalidArgument;
  *asyncError = ncclSuccess;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) return ncclInvalidArgument;
  *count = comm->comm->bootstrap()->getNranks();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) return ncclInvalidArgument;
  *device = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return ncclInvalidArgument;
  *rank = comm->comm->bootstrap()->getRank();
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*, ncclDataType_t, ncclScalarResidence_t, ncclComm_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t,
                                 cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBcast(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBroadcast(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t reductionOperation, ncclComm_t comm, cudaStream_t stream) {
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr)
    return ncclInvalidArgument;

  // Declarating variables
  size_t bytes = count * ncclTypeSize(datatype);
  int rank = comm->comm->bootstrap()->getRank();

  if (bytes < comm->allReduceSmallMessageSizeBoundary) {
    return ncclAllReduceFallback(sendbuff, recvbuff, count, datatype, reductionOperation, comm, stream);
  } else {
    std::shared_ptr<mscclpp::ExecutionPlan> plan;
    if (bytes <= comm->allReduceLargeMessageSizeBoundary)
      plan = (sendbuff == recvbuff) ? comm->allReducePacketIPPlan : comm->allReducePacketOPPlan;
    else {
      plan = (sendbuff == recvbuff) ? comm->allReduceIPPlan : comm->allReduceOPPlan;
    }

    if (plan == nullptr)
      return ncclAllReduceFallback(sendbuff, recvbuff, count, datatype, reductionOperation, comm, stream);

    switch (datatype) {
      case ncclFloat16:
        comm->executor->execute(rank, (half*)sendbuff, (half*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT16, *plan,
                                stream, mscclpp::PacketType::LL8);
        break;
      case ncclFloat32:
        comm->executor->execute(rank, (float*)sendbuff, (float*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT32,
                                *plan, stream, mscclpp::PacketType::LL8);
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
        return ncclInvalidArgument;
    }
  }

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                        cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;

  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();

  if (bytes * nRank < comm->allGatherSmallMessageSizeBoundary)
    return ncclAllGatherFallback(sendbuff, recvbuff, sendcount, datatype, comm, stream);

  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  if (bytes * nRank <= comm->allGatherLargeMessageSizeBoundary)
    plan = (sendbuff == (char*)recvbuff + rank * bytes) ? comm->allGatherPacketIPPlan : comm->allGatherPacketOPPlan;
  else {
    plan = (sendbuff == (char*)recvbuff + rank * bytes) ? comm->allGatherIPPlan : comm->allGatherOPPlan;
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
      return ncclInvalidArgument;
  }

  return ncclSuccess;
}

NCCL_API ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
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
