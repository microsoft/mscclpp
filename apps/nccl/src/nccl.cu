// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
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
__forceinline__ __device__ uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
__forceinline__ __device__ uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (size_t i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<T>(dst4[i], src4[i]);
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*)dst) + nInt4 * 4;
    int* srcLast = ((int*)src) + nInt4 * 4;
    for (size_t i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
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

struct ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  std::unordered_map<channelKey, std::vector<mscclpp::SmChannel>> smChannels;
  std::unordered_map<channelKey, std::vector<mscclpp::SmChannel>> smOutChannels;
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
__global__ void allreduce1(T* src, T* dst, int rank, int nranks, size_t nelems) {
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
    constSmChannels[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    constSmChannels[tid - nPeer].wait();
  }
  deviceSyncer.sync(gridDim.x);

  // use int4 as much as possible
  const size_t nInt4 = chunkSize / vectorSize;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * gridDim.x) {
    int4 tmp = src4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = constSmChannels[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[indexOffset4 + idx] = tmp;
  }

  // use the given TYPE for the rest
  size_t processed = nInt4 * vectorSize * nranks;
  const size_t nRemElems = nelems - processed;
  const size_t startIdx = processed + (nRemElems * rank) / nranks;
  const size_t endIdx = processed + (nRemElems * (rank + 1)) / nranks;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x + startIdx; idx < endIdx; idx += blockDim.x * gridDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nPeer; ++index) {
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      T val = constSmChannels[peerIdx].read<T>(idx);
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
    constSmChannels[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    constSmChannels[tid - nPeer].wait();
  }

  deviceSyncer.sync(gridDim.x);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank);
    if (peerIdx >= nPeer) peerIdx -= nPeer;
    const int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = chunkSize * remoteRank * sizeof(T);
    constSmOutChannels[peerIdx].get(offset, chunkSize * sizeof(T), tid, blockDim.x * gridDim.x);
  }
}

template <typename T>
__global__ void allreduce7(T* buff, T* scratch, T* resultBuff, int rank, int nRanksPerNode, int worldSize,
                           size_t nelems) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  nelems = nelems / (sizeof(int) / sizeof(T));
  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LL8Packet);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LL8Packet) : 3 * nPkts * sizeof(mscclpp::LL8Packet);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint32_t* src = (uint32_t*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint32_t* dst = (uint32_t*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // step 1: write to scratch buffer
  constSmChannels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                                                          blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint32_t data = 0;
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nPktsPerRank;
      uint32_t val = dstPkt[idx].read(flag);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LL8Packet packet;
    packet.data = data;
    packet.flag = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LL8Packet) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      constSmChannels[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint32_t* result = (uint32_t*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint32_t data = dstPkt[idx + dstOffset].read(flag);
    result[idx] = data;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

template <typename T>
cudaError_t allreduce(T* buff, T* scratch, T* resultBuff, int rank, int nRanksPerNode, int worldSize, size_t nelems,
                      cudaStream_t stream) {
  if (sizeof(T) * nelems <= (1 << 20)) {
#if defined(__HIP_PLATFORM_AMD__)
    int nBlocks = 28;
    int nThreadsPerBlock = 1024;
    if (nelems >= 8192) {
      nBlocks = 56;
      nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
    }
    allreduce7<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, rank, nRanksPerNode, worldSize,
                                                         nelems);
#else
    allreduce6<<<21, 512, 0, stream>>>(buff, scratch, resultBuff, rank, nRanksPerNode, worldSize, nelems);
#endif
  } else {
    allreduce1<<<24, 1024, 0, stream>>>(buff, resultBuff, rank, worldSize, nelems);
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
  for (size_t cid = 0; cid < comm->connections.size(); ++cid) {
    if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      channels.emplace_back(smSemaphores[cid], remoteMemories[cid], src, nullptr);
    }
  }
  return channels;
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

NCCL_API ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
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
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      smSemaphores.emplace_back(
          std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
    }
  }
  mscclppComm->setup();

  ncclComm* commPtr = new ncclComm();
  commPtr->comm = mscclppComm;
  commPtr->connections = std::move(connections);
  commPtr->smSemaphores = std::move(smSemaphores);
  // using scratch buffer for message size less then 1MB
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(scratchSize);
  commPtr->remoteScratchRegMemories = setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), scratchSize,
                                                          mscclpp::Transport::CudaIpc | IBs[rank % nRanksPerNode]);

  *comm = commPtr;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  delete comm;
  return ncclSuccess;
}

NCCL_API ncclResult_t ncclCommAbort(ncclComm_t comm) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config) {
  // TODO: implement this function
  return ncclInternalError;
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

NCCL_API const char* ncclGetLastError(ncclComm_t comm) {
  // TODO: implement this function
  return nullptr;
}

NCCL_API ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  // TODO: implement this function
  return ncclInternalError;

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

NCCL_API ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = count * ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  channelKey key{sendbuff, recvbuff, bytes};
  if (bytes <= 1 << 20) {
    auto it = comm->smChannels.find(key);
    if (it == comm->smChannels.end()) {
      std::vector<mscclpp::SmChannel> channels =
          setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>(sendbuff));
      it = comm->smChannels.emplace(key, channels).first;
    }
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(it->second.begin(), it->second.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    // TODO: if sendbuff and recvbuff don't change, we can avoid copying smChannelDeviceHandles to device
    CUDACHECK(cudaMemcpyToSymbol(constSmChannels, smChannelDeviceHandles.data(),
                                 sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size()));
  } else {
    auto it = comm->smChannels.find(key);
    auto outIt = comm->smOutChannels.find(key);
    if (it == comm->smChannels.end()) {
      std::vector<mscclpp::RegisteredMemory> remoteMemories =
          setupRemoteMemories(comm->comm, rank, const_cast<void*>(sendbuff), bytes,
                              mscclpp::Transport::CudaIpc | IBs[rank % nRanksPerNode]);
      std::vector<mscclpp::SmChannel> channels = setupSmChannels(comm, remoteMemories, const_cast<void*>(sendbuff));
      it = comm->smChannels.emplace(key, channels).first;
      if (sendbuff != recvbuff) {
        std::vector<mscclpp::RegisteredMemory> remoteMemories =
            setupRemoteMemories(comm->comm, rank, recvbuff, bytes, mscclpp::Transport::CudaIpc | IBs[rank % nRanksPerNode]);
        std::vector<mscclpp::SmChannel> outChannels = setupSmChannels(comm, remoteMemories, recvbuff);
        outIt = comm->smOutChannels.emplace(key, outChannels).first;
      }
    }
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(it->second.begin(), it->second.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    // TODO: if sendbuff and recvbuff don't change, we can avoid copying smChannelDeviceHandles to device
    CUDACHECK(cudaMemcpyToSymbol(constSmChannels, smChannelDeviceHandles.data(),
                                 sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size()));
    if (sendbuff != recvbuff) {
      smChannelDeviceHandles.clear();
      std::transform(outIt->second.begin(), outIt->second.end(), std::back_inserter(smChannelDeviceHandles),
                     [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    }
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

NCCL_API ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  // TODO: implement this function
  return ncclInternalError;
}

NCCL_API ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
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
