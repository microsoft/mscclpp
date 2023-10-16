// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency.hpp>
#include <mscclpp/sm_channel_device.hpp>

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;

__device__ void localReduceScatterSm2(mscclpp::SmChannelDeviceHandle* smChans, int* buff, int* scratch, int rank,
                                      int nRanksPerNode, size_t chunkSize, size_t nelems, int nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = localRankIndexInNode * chunkSize;
  const size_t indexOffset4 = indexOffset / 4;

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

  const size_t nInt4 = nelems / 4;
  for (int index = 0; index < nPeer; ++index) {
    int4 val;
    int peerIdx = (index + localRankIndexInNode) % nPeer;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * nBlocks) {
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      buff4[indexOffset4 + idx].w += val.w;
      buff4[indexOffset4 + idx].x += val.x;
      buff4[indexOffset4 + idx].y += val.y;
      buff4[indexOffset4 + idx].z += val.z;
    }
  }

  const size_t nLastInts = nelems % 4;
  for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nLastInts; idx += blockDim.x * nBlocks) {
      int val = smChans[(localRankIndexInNode + peerIdx) % nPeer].read<int>(indexOffset + nInt4 * 4 + idx);
      buff[indexOffset + nInt4 * 4 + idx] += val;
    }
  }
}

__device__ void localRingAllGatherSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode,
                                     uint64_t size, size_t nBlocks) {
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
    int peerIdx = (i + rank) % nPeer;
    const int remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex;
    smChans[peerIdx].get(offset, size, tid, blockDim.x * nBlocks);
  }
}

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce1(mscclpp::SmChannelDeviceHandle* smChans, int* buff, int rank, int nranks, int nelems) {
  int* scratch = buff + nelems;
  localReduceScatterSm2(smChans, buff, scratch, rank, nranks, nelems / nranks, nelems / nranks, gridDim.x);
  deviceSyncer.sync(gridDim.x);
  localRingAllGatherSm(smChans, rank, nranks, nelems / nranks * sizeof(int), gridDim.x);
}
