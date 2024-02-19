// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <map>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <vector>

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

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (int i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<T>(dst4[i], src4[i]);
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*)dst) + nInt4 * 4;
    int* srcLast = ((int*)src) + nInt4 * 4;
    for (int i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
      dstLast[i] = add_vectors<T>(dstLast[i], srcLast[i]);
    }
  }
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

// TODO:
static const int nRanksPerNode = 8;
// Only use scratch buffer for message size less then 1MB
static const int scratchSize = 1024 * 1024 * 8;

static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                            mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                            mscclpp::Transport::IB6, mscclpp::Transport::IB7};

__constant__ mscclpp::DeviceHandle<mscclpp::SmChannel> constSmChannels[8];
__constant__ mscclpp::DeviceHandle<mscclpp::SmChannel> constSmOutChannels[8];
__device__ mscclpp::DeviceSyncer deviceSyncer;

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  // key is the pair of sendbuff and recvbuff
  std::map<std::pair<const void*, const void*>, std::vector<mscclpp::SmChannel>> smChannels;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
};

cudaError_t allreduce(int* buff, int* scratch, void* resultBuff, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, cudaStream_t stream);

#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/packet_device.hpp>

// extern __constant__ mscclpp::SmChannelDeviceHandle *constSmChannels;
__device__ uint64_t globalFlag;

template <typename T>
__global__ void allreduce6(T* buff, T* scratch, T* resultBuff, int rank, int nRanksPerNode, int worldSize,
                           size_t nelems) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  nelems = nelems / (sizeof(int) / sizeof(T));
  const int nPeers = nRanksPerNode - 1;
  const int nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag + 1;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = constSmChannels[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // step 1: write to scratch buffer
  smChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx].x = data.x;
    dst[idx].y = data.y;
    for (int index = 0; index < nPeers; index++) {
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)constSmChannels[index].dst_ + scratchResultOffset);
      dstPkt[idx + rank * nPktsPerRank].write(data.x, data.y, flag);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

template <typename T>
__global__ void allreduce1(mscclpp::SmChannelDeviceHandle* smChans, mscclpp::SmChannelDeviceHandle* smOutChans, T* src,
                           T* dst, int rank, int nranks, size_t nelems) {
  const size_t chunkSize = nelems / nranks;
  if (nranks == 1) return;
  const int nPeer = nranks - 1;
  const size_t indexOffset = rank * chunkSize;
  const size_t vectorSize = sizeof(int4) / sizeof(T);
  const size_t indexOffset4 = indexOffset / vectorSize;
  int4* src4 = (int4*)src;
  int4* dst4 = (int4*)dst;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // synchronize everyone
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChans[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChans[tid - nPeer].wait();
  }
  deviceSyncer.sync(gridDim.x);

  // use int4 as much as possible
  const size_t nInt4 = chunkSize / vectorSize;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * gridDim.x) {
    int4 tmp = src4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[indexOffset4 + idx] = tmp;
  }

  // use the given TYPE for the rest
  size_t processed = nInt4 * vectorSize * nranks;
  const size_t nRemElems = nelems - processed;
  const size_t startIdx = processed + (nRemElems * rank) / nranks;
  const size_t endIdx = processed + (nRemElems * (rank + 1)) / nranks;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x + startIdx; idx < endIdx; idx += blockDim.x * gridDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nPeer; ++index) {
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      T val = smChans[peerIdx].read<T>(idx);
      tmp += val;
    }
    dst[idx] = tmp;
  }

  // synchronize everyone again
  deviceSyncer.sync(gridDim.x);
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChans[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChans[tid - nPeer].wait();
  }

  deviceSyncer.sync(gridDim.x);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank);
    if (peerIdx >= nPeer) peerIdx -= nPeer;
    const int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = chunkSize * remoteRank * sizeof(T);
    smOutChans[peerIdx].get(offset, chunkSize * sizeof(T), tid, blockDim.x * gridDim.x);
  }
}

template <typename T>
cudaError_t allreduce(T* buff, T* scratch, T* resultBuff, int rank, int nRanksPerNode, int worldSize, size_t nelems,
                      cudaStream_t stream) {
  if (sizeof(T) * nelems <= (1 << 20)) {
    allreduce6<<<21, 512, 0, stream>>>(buff, scratch, resultBuff, rank, nRanksPerNode, worldSize, nelems);
  } else {
    allreduce1<<<24, 1024, 0, stream>>>(constSmChannels, constSmOutChannels, buff, resultBuff, rank, worldSize, nelems);
  }
  return cudaGetLastError();
}

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
#endif // defined(__CUDA_BF16_TYPES_EXIST__)
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFp8E4M3:
    case ncclFp8E5M2:
      return 1;
#endif // defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclNumTypes:
      return 0;
  }
  return 0;
}

static mscclpp::Transport getTransport(int rank, int peerRank) {
  if (rank / nRanksPerNode == peerRank / nRanksPerNode) {
    return mscclpp::Transport::CudaIpc;
  } else {
    return IBs[rank % nRanksPerNode];
  }
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
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      smSemaphores.emplace_back(
          std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
    }
  }
  mscclppComm->setup();

  ncclComm* commPtr = new ncclComm();
  commPtr->comm = mscclppComm;
  commPtr->connections = connections;
  commPtr->smSemaphores = smSemaphores;
  // using scratch buffer for message size less then 1MB
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(scratchSize);

  mscclpp::RegisteredMemory memory = mscclppComm->registerMemory(
      commPtr->scratchBuff.get(), scratchSize, mscclpp::Transport::CudaIpc | IBs[rank % nRanksPerNode]);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
  for (int i = 0; i < commPtr->comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    mscclpp::Transport transport = getTransport(rank, i);
    remoteRegMemoryFutures.push_back(commPtr->comm->recvMemoryOnSetup(i, 0));
    commPtr->comm->sendMemoryOnSetup(memory, i, 0);
  }
  commPtr->comm->setup();
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(),
                 std::back_inserter(commPtr->remoteScratchRegMemories),
                 [](const auto& future) { return future.get(); });
  *comm = commPtr;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  delete comm;
  return ncclSuccess;
}

NCCL_API const char* ncclGetErrorString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  std::pair<const void*, const void*> key(sendbuff, recvbuff);
  if (bytes <= 1 << 20) {
    std::vector<mscclpp::SmChannel> channels;
    if (comm->smChannels.find(key) == comm->smChannels.end()) {
      std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores = comm->smSemaphores;
      for (size_t cid = 0; cid < comm->connections.size(); ++cid) {
        if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
          channels.emplace_back(smSemaphores[cid], comm->remoteScratchRegMemories[cid], const_cast<void*>(sendbuff),
                                nullptr);
        }
      }
      comm->smChannels.emplace(key, channels);
    } else {
      channels = comm->smChannels[key];
    }
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    // TODO: if sendbuff and recvbuff don't change, we can avoid copying smChannelDeviceHandles to device
    CUDACHECK(cudaMemcpyToSymbol(constSmChannels, smChannelDeviceHandles.data(),
                                 sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size()));
  } else {
    // TODO: Debug this
    std::vector<mscclpp::SmChannel> channels;
    if (comm->smChannels.find(key) == comm->smChannels.end()) {
      mscclpp::RegisteredMemory memory = comm->comm->registerMemory(
          const_cast<void*>(sendbuff), bytes, mscclpp::Transport::CudaIpc | IBs[rank % nRanksPerNode]);
      std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
      for (int i = 0; i < comm->comm->bootstrap()->getNranks(); i++) {
        if (i == rank) continue;
        mscclpp::Transport transport = getTransport(rank, i);
        remoteRegMemoryFutures.push_back(comm->comm->recvMemoryOnSetup(i, 0));
        comm->comm->sendMemoryOnSetup(memory, i, 0);
      }
      comm->comm->setup();
      std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores = comm->smSemaphores;
      for (size_t cid = 0; cid < comm->connections.size(); ++cid) {
        if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
          channels.emplace_back(smSemaphores[cid], remoteRegMemoryFutures[cid].get(), const_cast<void*>(sendbuff),
                                nullptr);
        }
      }
      comm->smChannels.emplace(key, channels);
    } else {
      channels = comm->smChannels[key];
    }
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    // TODO: if sendbuff and recvbuff don't change, we can avoid copying smChannelDeviceHandles to device
    CUDACHECK(cudaMemcpyToSymbol(constSmChannels, smChannelDeviceHandles.data(),
                                 sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size()));
    CUDACHECK(cudaMemcpyToSymbol(constSmOutChannels, smChannelDeviceHandles.data(),
                                 sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size()));
  }

  switch (datatype) {
    case ncclFloat16:
      CUDACHECK(allreduce((half*)sendbuff, (half*)comm->scratchBuff.get(), (half*)recvbuff,
                          rank, nRanksPerNode, comm->comm->bootstrap()->getNranks(),
                          count, stream));
      break;
    case ncclFloat32:
      CUDACHECK(allreduce((float*)sendbuff, (float*)comm->scratchBuff.get(), (float*)recvbuff,
                          comm->comm->bootstrap()->getRank(), nRanksPerNode, comm->comm->bootstrap()->getNranks(),
                          count, stream));
      break;
    case ncclInt32:
    case ncclUint32:
      CUDACHECK(allreduce((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff,
                          comm->comm->bootstrap()->getRank(), nRanksPerNode, comm->comm->bootstrap()->getNranks(),
                          count, stream));
      break;
    default:
      return ncclInvalidArgument;
  }
  return ncclSuccess;
}
