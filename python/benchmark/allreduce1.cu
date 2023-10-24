// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency.hpp>
#include <mscclpp/sm_channel_device.hpp>
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
