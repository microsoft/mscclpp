// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <cuda_fp16.h>


__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;

#define VECTOR_SIZE (sizeof(int4) / sizeof(TYPE))

template<typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
    static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

    union {
        From f;
        To   t;
    } u;
    u.f = src;
    return u.t;
}

template<typename T>
__forceinline__ __device__ T add_elements(T a, T b){
  return a + b;
}

template<>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b){
  return __hadd2(a, b);
}

template<typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template<typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a,b);
}

template<>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template<typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template<typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a,b);
}

template<>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}


template<typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template<typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a,b);
}

template<>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}


__forceinline__ __device__ void vectorSum(TYPE* dst, TYPE* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (int i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<TYPE>(dst4[i], src4[i]);
    // dst4[i].w += src4[i].w;
    // dst4[i].x += src4[i].x;
    // dst4[i].y += src4[i].y;
    // dst4[i].z += src4[i].z;
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*) dst) + nInt4 * 4;
    int* srcLast = ((int*) src) + nInt4 * 4;
    for (int i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
      dstLast[i] = add_vectors<TYPE>(dstLast[i], srcLast[i]);
    }
  }
}

__forceinline__ __device__ void vectorSum(TYPE* dst, TYPE* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}


// AllReduce1
// -------------------------------------------

__device__ void localReduceScatterSm2(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank,
                                      int nRanksPerNode, size_t chunkSize, size_t nelems, int nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = localRankIndexInNode * chunkSize;
  const size_t indexOffset4 = indexOffset / VECTOR_SIZE;

  int4* buff4 = (int4*)buff;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nPeer) {
    smChans[tid].signal();
  }
  const int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < nBlocks * blockDim.x) {
    smChans[tid - waitStart].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);

  const size_t nInt4 = nelems / VECTOR_SIZE;
  for (int index = 0; index < nPeer; ++index) {
    int4 val;
    int peerIdx = (index + localRankIndexInNode);
    if (peerIdx >= nPeer) peerIdx -= nPeer;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * nBlocks) {
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      buff4[indexOffset4 + idx] = add_vectors<TYPE>(buff4[indexOffset4 + idx], val);
    }
  }

  const size_t nLastInts = nelems % VECTOR_SIZE;
  for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nLastInts; idx += blockDim.x * nBlocks) {
      int val = smChans[(localRankIndexInNode + peerIdx) % nPeer].read<int>(indexOffset + nInt4 * VECTOR_SIZE + idx);
      buff[indexOffset + nInt4 * VECTOR_SIZE + idx] += val;
    }
  }
}

__device__ void localRingAllGatherSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode, uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nPeer = nRanksPerNode - 1;

  if (tid < nPeer) {
    smChans[tid].signal();
  }
  int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < nBlocks * blockDim.x) {
    smChans[tid - waitStart].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank);
    if (peerIdx >= nPeer) peerIdx -= nPeer;
    const int remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex;
    smChans[peerIdx].get(offset, size, tid, blockDim.x * nBlocks);
  }
}

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce1(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank, int nranks, int nelems) {
  localReduceScatterSm2(smChans, buff, rank, nranks, nelems / nranks, nelems / nranks, gridDim.x);
  deviceSyncer.sync(gridDim.x);
  localRingAllGatherSm(smChans, rank, nranks, nelems / nranks * sizeof(TYPE), gridDim.x);
}

// AllReduce2
// -------------------------------------------

__device__ uint64_t globalFlag = 1;

extern "C" __global__ void __launch_bounds__(512, 1)
allreduce2(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, TYPE* scratch, void* resultBuff, int rank, int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  // This version of allreduce only works for single nodes
  const int nPeers = worldSize - 1;
  const int nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  // flag for packets. Initially 1
  const uint32_t flag = (uint32_t)globalFlag;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChans[peerIdx];
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
      data = add_vectors<TYPE>(val, data);
    }
    data = add_vectors<TYPE>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      smChans[index].write(offset, packet);
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

// AllReduce3
// -------------------------------------------

extern "C" __global__ void __launch_bounds__(1024, 1)
allreduce3(mscclpp::SimpleProxyChannelDeviceHandle* fstRoundChans, mscclpp::SimpleProxyChannelDeviceHandle* sndRoundChans, TYPE* buff, TYPE* scratch, int rank, int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));

  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::SimpleProxyChannelDeviceHandle& devFstSendChan = fstRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devFstRecvChan = fstRoundChans[peerRecvId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndSendChan = sndRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndRecvChan = sndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkNelem = nelems / worldSize;
  size_t offset = chunkIndex * chunkNelem * sizeof(int);
  if (isComm) {
    if (chunkNelem > 1) {
      devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }

  // Step 2 ~ Step n-1
  for (int step = 2; step < worldSize; ++step) {
    if (isComm) {
      if (chunkNelem > 1) {
        devFstRecvChan.wait();
        devFstSendChan.flush();
      }
      devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - step) % worldSize;
    offset = chunkIndex * chunkNelem * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

    if (isComm) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
      if (chunkNelem > 1) {
        devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
      }
    }
    deviceSyncer.sync(gridDim.x);

    dst += chunkNelem / 2;
    src += chunkNelem / 2;
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);
  }

  // Step n
  if (isComm) {
    if (chunkNelem > 1) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
    }
    devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
  }
  deviceSyncer.sync(gridDim.x);

  offset = rank * chunkNelem * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

  if (isComm) {
    devFstRecvChan.wait();
    devFstSendChan.flush();
    if (chunkNelem > 1) {
      devSndSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }
  deviceSyncer.sync(gridDim.x);

  dst += chunkNelem / 2;
  src += chunkNelem / 2;
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);

  if (isComm) {
    if (chunkNelem > 1) {
      devSndRecvChan.wait();
      devSndSendChan.flush();
    }
    devSndSendChan.putWithSignalAndFlush(offset + chunkNelem / 2 * sizeof(int),
                                         (chunkNelem - chunkNelem / 2) * sizeof(int));
  }

  // Step n+1 ~ Step 2n-2
  for (int i = 1; i < worldSize - 1; ++i) {
    if (isComm) {
      devSndRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Copy
    chunkIndex = (rank + worldSize - i) % worldSize;
    if (isComm) {
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkNelem * sizeof(int), chunkNelem * sizeof(int));
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}
