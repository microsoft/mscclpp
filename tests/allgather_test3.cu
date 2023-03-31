/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "cuda_runtime.h"

#include <string>

#define ALIGN 4

__device__ void allgather0(mscclppDevConn_t devConn, int rank, int world_size, int remoteRank, int nelemsPerGPU)
{
  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  devConn.put(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  devConn.signal();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  devConn.wait();
}

__device__ void allgather1(mscclppDevConn_t devConn, int rank, int world_size, int remoteRank, int nelemsPerGPU)
{
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % world_size
  // Step 2: GPU rank i waits for data from GPU rank (i+2) % world_size
  // ...
  // This order is much better for DMA engine for NVLinks

  for (int i = 1; i < world_size; i++) {
    __syncthreads();
    if (remoteRank != ((rank + i) % world_size))
      continue;
    // put your data to GPU (rank+i) % world_size and signal all in one call
    devConn.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  }
  // all connections wait for the signal from the sender
  devConn.wait();
}

__global__ void kernel(int rank, int world_size, int nelemsPerGPU, int kernel)
{
  // only use a single thread from each warp
  if (threadIdx.x % 32 != 0)
    return;

  // find the mapping between remoteRank and devConns
  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  // Each warp is responsible for one of the remote ranks
  mscclppDevConn_t devConn = constDevConns[warpId];

  if (kernel == 0)
    allgather0(devConn, rank, world_size, remoteRank, nelemsPerGPU);
  else if (kernel == 1)
    allgather1(devConn, rank, world_size, remoteRank, nelemsPerGPU);
}

void AllGatherGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                               size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / (ALIGN * nranks)) * ALIGN;
  *sendcount = base;
  *recvcount = base * nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, int in_place) {
  size_t sendcount = args->sendBytes;
  size_t recvcount = args->expectedBytes;
  int nranks = args->totalProcs;

  CUDACHECK(cudaSetDevice(args->gpus[0]));
  int rank = args->proc;
  CUDACHECK(cudaMemset(args->recvbuffs[0], 0, args->expectedBytes));
  void* data = in_place ? ((char*)args->recvbuffs[0]) + rank * args->sendBytes : args->sendbuffs[0];

  int* dataHost = new int[recvcount];
  for (int i = 0; i < static_cast<int>(recvcount); i++) {
    int val = i + 1;
    if (i / args->ranksPerNode == rank) {
      dataHost[i] = val;
    } else {
      dataHost[i] = 0;
    }
  }
  CUDACHECK(cudaMemcpy(args->recvbuffs[0], dataHost, recvcount, cudaMemcpyHostToDevice));
  delete dataHost;
  // TODO: need to init expected data here
  CUDACHECK(cudaDeviceSynchronize());

  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t AllGatherRunColl(void* sendbuff, void* recvbuff, size_t count)
{
  // NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count, type, comm, stream));
  return testSuccess;
}

struct testColl allGatherTest = {"AllGather", AllGatherGetCollByteCount, AllGatherInitData, AllGatherGetBw,
                                 AllGatherRunColl};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args)
{
  args->collTest = &allGatherTest;

  TESTCHECK(TimeTest(args));
  return testSuccess;
}

struct testEngine allGatherEngine = {AllGatherGetBuffSize, AllGatherRunTest};

#pragma weak mscclppTestEngine = allGatherEngine
