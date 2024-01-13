// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.cuh"

__device__ mscclpp::DeviceSyncer deviceSyncer;

// -------------------------------------------
// AllReduce1
// -------------------------------------------

template <int READ_ONLY>
__device__ void allreduce1_helper(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank, int nranks,
                                  size_t nelems) {
  const size_t chunkSize = nelems / nranks;
  if (nranks == 1) return;
  const int nPeer = nranks - 1;
  const size_t indexOffset = rank * chunkSize;
  const size_t indexOffset4 = indexOffset / VECTOR_SIZE;
  int4* buff4 = (int4*)buff;
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
  const size_t nInt4 = chunkSize / VECTOR_SIZE;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * gridDim.x) {
    int4 tmp = buff4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<TYPE>(tmp, val);
    }
    if (READ_ONLY == 0) {
      for (int index = 0; index < nPeer; ++index) {
        int peerIdx = (index + rank);
        if (peerIdx >= nPeer) peerIdx -= nPeer;
        smChans[peerIdx].write<int4>(indexOffset4 + idx, tmp);
      }
    }
    buff4[indexOffset4 + idx] = tmp;
  }

  // use the given TYPE for the rest
  size_t processed = nInt4 * VECTOR_SIZE * nranks;
  const size_t nRemElems = nelems - processed;
  const size_t startIdx = processed + (nRemElems * rank) / nranks;
  const size_t endIdx = processed + (nRemElems * (rank + 1)) / nranks;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x + startIdx; idx < endIdx; idx += blockDim.x * gridDim.x) {
    TYPE tmp = buff[idx];
    for (int index = 0; index < nPeer; ++index) {
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      TYPE val = smChans[peerIdx].read<TYPE>(idx);
      tmp += val;
    }
    if (READ_ONLY == 0) {
      for (int index = 0; index < nPeer; ++index) {
        int peerIdx = (index + rank);
        if (peerIdx >= nPeer) peerIdx -= nPeer;
        smChans[peerIdx].write<TYPE>(idx, tmp);
      }
    }
    buff[idx] = tmp;
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

  if (READ_ONLY) {
    deviceSyncer.sync(gridDim.x);
    for (int i = 0; i < nPeer; ++i) {
      int peerIdx = (i + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      const int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
      size_t offset = chunkSize * remoteRank * sizeof(TYPE);
      smChans[peerIdx].get(offset, chunkSize * sizeof(TYPE), tid, blockDim.x * gridDim.x);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1) allreduce1(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff,
                                                                 int rank, int nranks, size_t nelems, int read_only) {
  if (read_only)
    allreduce1_helper<1>(smChans, buff, rank, nranks, nelems);
  else
    allreduce1_helper<0>(smChans, buff, rank, nranks, nelems);
}
