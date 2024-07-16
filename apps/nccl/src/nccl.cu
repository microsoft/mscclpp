// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/executor.hpp>
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
  bool operator==(const channelKey& other) const {
    return buff == other.buff && bytes == other.bytes;
  }
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
  std::shared_ptr<mscclpp::Executor> executor;
  std::shared_ptr<mscclpp::ExecutionPlan> allReduceMI300SmPlanIP, allReduceMI300PacketPlanIP, allReduceMI300PacketSmallSizePlanIP;
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

  ncclComm* commPtr = new ncclComm();
  commPtr->comm = mscclppComm;
  commPtr->executor = std::make_shared<mscclpp::Executor>(mscclppComm);
  commPtr->allReduceMI300SmPlanIP = std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allreduce_mi300", "/root/mscclpp/apps/nccl/test/execution-files/allreducesm.json"));
  commPtr->allReduceMI300PacketPlanIP = std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allreduce_mi300_packet", "/root/mscclpp/apps/nccl/test/execution-files/allreducepacket.json"));
  commPtr->allReduceMI300PacketSmallSizePlanIP = std::make_shared<mscclpp::ExecutionPlan>(mscclpp::ExecutionPlan("allreduce_mi300_packet", "/root/mscclpp/apps/nccl/test/execution-files/allreducepacket_smallsize.json"));

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
  // Checking if the parameters are valids
  if (sendbuff == nullptr || recvbuff == nullptr || count == 0 || ncclTypeSize(datatype) == 0 || comm == nullptr) return ncclInvalidArgument;

  // Declarating variables
  size_t bytes = count * ncclTypeSize(datatype);
  int rank = comm->comm->bootstrap()->getRank();

  std::shared_ptr<mscclpp::ExecutionPlan> plan;
  if(bytes <= 4 * (1 << 10)) plan = sendbuff == recvbuff ? comm->allReduceMI300PacketSmallSizePlanIP : nullptr;
  else if(bytes <= (1 << 20)) plan = sendbuff == recvbuff ? comm->allReduceMI300PacketPlanIP : nullptr;
  else plan = sendbuff == recvbuff ? comm->allReduceMI300SmPlanIP : nullptr;

  switch (datatype) {
    case ncclFloat16:
      comm->executor->execute(rank, (half*)sendbuff, (half*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT16, 32,
                      *plan, stream, mscclpp::PacketType::LL8);
      break;
    case ncclFloat32:
      comm->executor->execute(rank, (float*)sendbuff, (float*)recvbuff, bytes, bytes, mscclpp::DataType::FLOAT32, 32,
                      *plan, stream, mscclpp::PacketType::LL8);
      break;
    case ncclInt32:
    case ncclUint32:
      comm->executor->execute(rank, (int*)sendbuff, (int*)recvbuff, bytes, bytes, mscclpp::DataType::UINT32, 32,
                      *plan, stream, mscclpp::PacketType::LL8);
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
  /* size_t bytes = sendcount * ncclTypeSize(datatype);
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
    std::vector<mscclpp::RegisteredMemory> remoteMemories =
        setupRemoteMemories(comm->comm, rank, const_cast<void*>((void*)recvBasePtr), recvBytes,
                            mscclpp::Transport::CudaIpc);
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
    CUDACHECK(allgather<false>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, offsetOut,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  } else {
    CUDACHECK(allgather<true>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, offsetOut,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  }
  return ncclSuccess; */
  return ncclInternalError;
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
