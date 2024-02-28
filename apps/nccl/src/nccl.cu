// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
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
__device__ mscclpp::DeviceSyncer deviceSyncer;

// static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
//                             mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
//                             mscclpp::Transport::IB6, mscclpp::Transport::IB7};

struct channelKey {
  const void* sendbuff;
  const void* recvbuff;
  size_t bytes;
  bool operator==(const channelKey& other) const {
    return sendbuff == other.sendbuff && recvbuff == other.recvbuff && bytes == other.bytes;
  }
};

namespace std {
template <>
struct hash<channelKey> {
  std::size_t operator()(const channelKey& k) const {
    return std::hash<const void*>()(k.sendbuff) ^ std::hash<const void*>()(k.recvbuff) ^ std::hash<size_t>()(k.bytes);
  }
};
}  // namespace std

struct ChannelInfo {
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SmChannel> smOutChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smOutChannelDeviceHandles;
};

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  std::unordered_map<channelKey, ChannelInfo> channelInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
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

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
                 [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(smChannelDeviceHandles.size());
  mscclpp::AvoidCudaGraphCaptureGuard guard;
  CUDACHECK(cudaMemcpy(ptr.get(), smChannelDeviceHandles.data(),
                       sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size(),
                       cudaMemcpyHostToDevice));
  return ptr;
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

NCCL_API ncclResult_t ncclCommInitRankConfig(ncclComm_t*, int, ncclUniqueId, int,
                                             ncclConfig_t*) {
  // TODO: implement this function
  return ncclInternalError;
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
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(SCRATCH_SIZE);
  commPtr->remoteScratchRegMemories =
      setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);

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
  return nullptr;
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

NCCL_API ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t*, void*, ncclDataType_t,
                                               ncclScalarResidence_t, ncclComm_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpDestroy(ncclRedOp_t, ncclComm_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclReduce(const void*, void*, size_t, ncclDataType_t,
                                 ncclRedOp_t, int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBcast(void*, size_t, ncclDataType_t, int, ncclComm_t,
                                cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBroadcast(const void*, void*, size_t, ncclDataType_t,
                                    int, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  channelKey key{sendbuff, recvbuff, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels = nullptr;

  auto it = comm->channelInfos.find(key);
  if (it == comm->channelInfos.end()) {
    // setup smChannels (src: sendbuff, dst: remote scratch buff)
    std::vector<mscclpp::SmChannel> channels = setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>(sendbuff));
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = comm->channelInfos.emplace(key, channelInfo).first;

    // setup smOutChannels (src: recvbuff, dst: remote recvbuff)
    if (bytes > (1 << 20)) {
      std::vector<mscclpp::RegisteredMemory> remoteMemories =
          setupRemoteMemories(comm->comm, rank, recvbuff, bytes, mscclpp::Transport::CudaIpc);
      std::vector<mscclpp::SmChannel> outChannels = setupSmChannels(comm, remoteMemories, recvbuff);
      it->second.smOutChannels = outChannels;
      it->second.smOutChannelDeviceHandles = setupSmChannelDeviceHandles(outChannels);
    }
  }

  smChannels = it->second.smChannelDeviceHandles.get();
  smOutChannels = it->second.smOutChannelDeviceHandles.get();

  switch (datatype) {
    case ncclFloat16:
      CUDACHECK(allreduce((half*)sendbuff, (half*)comm->scratchBuff.get(), (half*)recvbuff, smChannels, smOutChannels,
                          rank, NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case ncclFloat32:
      CUDACHECK(allreduce((float*)sendbuff, (float*)comm->scratchBuff.get(), (float*)recvbuff, smChannels,
                          smOutChannels, comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case ncclInt32:
    case ncclUint32:
      CUDACHECK(allreduce((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, smOutChannels,
                          comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(),
                          count, stream));
      break;
    default:
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t,
                                        ncclRedOp_t, ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  channelKey key{sendbuff, recvbuff, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;

  auto it = comm->channelInfos.find(key);
  if (it == comm->channelInfos.end()) {
    std::vector<mscclpp::RegisteredMemory> remoteMemories =
        setupRemoteMemories(comm->comm, rank, const_cast<void*>(recvbuff), bytes * nRank,
                            mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(comm, remoteMemories, const_cast<void*>(recvbuff));
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = comm->channelInfos.emplace(key, channelInfo).first;
  }
  smChannels = it->second.smChannelDeviceHandles.get();
  if ((char*)sendbuff == (char*)recvbuff + rank * sendcount) {
    CUDACHECK(allgather<false>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  } else {
    CUDACHECK(allgather<true>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  }
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t,
                               cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t,
                               cudaStream_t) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t,
                                   ncclComm_t, cudaStream_t) {
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
