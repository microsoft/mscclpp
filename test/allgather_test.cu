#include "comm.h"
#include "common.h"

#include <cuda_runtime.h>
#include <string>

#define ALIGN 4
__constant__ mscclppDevConn_t constDevConns[16];

__device__ void allgather0(mscclppDevConn_t devConn, int rank, int world_size, int remoteRank, size_t nelemsPerGPU)
{
  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % 32 == 0)
    devConn.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % 32 == 0)
    devConn.flush();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % 32 == 0)
    devConn.wait();
}

__device__ void localAllGather(mscclppDevConn_t devConn, int rank, int world_size, int nranksPerNode, int remoteRank,
                               uint64_t offset, uint64_t size)
{
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nranksPerNode
  // and waits for data from GPU rank (i-1) % nranksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nranksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  for (int i = 1; i < nranksPerNode; i++) {
    if ((remoteRank % nranksPerNode) == ((rank + i) % nranksPerNode)) {
      // put your data to GPU (rank+i) % nranksPerNode and signal in one call
      if ((threadIdx.x % 32) == 0)
        devConn.putWithSignalAndFlush(offset, size);
    }
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    if ((remoteRank % nranksPerNode) == ((rank - i + nranksPerNode) % nranksPerNode)) {
      if ((threadIdx.x % 32) == 0)
        devConn.wait();
    }
    asm volatile("bar.sync %0, %1;" ::"r"(11), "r"((nranksPerNode - 1) * 32) : "memory");
  }
}

__device__ void allgather1(mscclppDevConn_t devConn, int rank, int world_size, int nranksPerNode, int remoteRank,
                           size_t nelemsPerGPU)
{
  localAllGather(devConn, rank, world_size, nranksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                 nelemsPerGPU * sizeof(int));
}

__device__ void allgather2(mscclppDevConn_t devConn, int rank, int world_size, int nranksPerNode, int remoteRank,
                           size_t nelemsPerGPU)
{
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;

  // Step 1
  // local allgather
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devConn, rank, world_size, nranksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                   nelemsPerGPU * sizeof(int));
  }
  // cross-node exchange
  if (remoteRank % nranksPerNode == rank % nranksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devConn.putWithSignalAndFlush(rank * nelemsPerGPU * sizeof(int),
                                    (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0)
      devConn.wait();
  }

  __syncthreads();

  // Step 2
  // local allgather
  int otherNghr = (rank + nranksPerNode) % world_size;
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devConn, rank, world_size, nranksPerNode, remoteRank, otherNghr * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  }

  // cross-node exchange
  if (remoteRank % nranksPerNode == rank % nranksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devConn.putWithSignalAndFlush((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) *
                                      sizeof(int),
                                    nelemsPerGPU / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0)
      devConn.wait();
  }

  __syncthreads();

  // Step 3
  // local allgather
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devConn, rank, world_size, nranksPerNode, remoteRank,
                   (otherNghr * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
  }
}

__global__ void kernel(int rank, int world_size, int nranksPerNode, size_t nelemsPerGPU, int kernel)
{
  // find the mapping between remoteRank and devConns
  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  // Each warp is responsible for one of the remote ranks
  mscclppDevConn_t devConn = constDevConns[warpId];

  if (kernel == 0)
    allgather0(devConn, rank, world_size, remoteRank, nelemsPerGPU);
  else if (kernel == 1)
    allgather1(devConn, rank, world_size, nranksPerNode, remoteRank, nelemsPerGPU);
  else if (kernel == 2)
    allgather2(devConn, rank, world_size, nranksPerNode, remoteRank, nelemsPerGPU);
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

testResult_t AllGatherInitData(struct testArgs* args, int in_place)
{
  size_t sendcount = args->sendBytes / sizeof(int);
  size_t recvcount = args->expectedBytes / sizeof(int);
  // int nranks = args->totalProcs;

  CUDACHECK(cudaSetDevice(args->gpuNum));
  int rank = args->proc;
  CUDACHECK(cudaMemset(args->recvbuff, 0, args->expectedBytes));
  // void* data = in_place ? ((char*)args->recvbuffs[0]) + rank * args->sendBytes : args->sendbuffs[0];

  int* dataHost = new int[recvcount];
  for (size_t i = 0; i < recvcount; i++) {
    int val = i + 1;
    if (i / sendcount == (size_t)rank) {
      dataHost[i] = val;
    } else {
      dataHost[i] = 0;
    }
  }
  CUDACHECK(cudaMemcpy(args->recvbuff, dataHost, recvcount * sizeof(int), cudaMemcpyHostToDevice));
  for (int i = 0; i < static_cast<int>(recvcount); i++) {
    dataHost[i] = i + 1;
  }
  CUDACHECK(cudaMemcpy(args->expected, dataHost, recvcount * sizeof(int), cudaMemcpyHostToDevice));
  delete dataHost;
  CUDACHECK(cudaDeviceSynchronize());
  MSCCLPPCHECK(mscclppBootstrapBarrier(args->comm));
  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks)
{
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t AllGatherRunColl(void* sendbuff, void* recvbuff, int nranksPerNode, size_t count, mscclppComm_t comm,
                              cudaStream_t stream, int kernel_num)
{
  int worldSize = comm->nRanks;
  kernel<<<1, 32 * (worldSize - 1), 0, stream>>>(comm->rank, worldSize, nranksPerNode, count / sizeof(int), kernel_num);
  return testSuccess;
}

struct testColl allGatherTest = {"AllGather",    AllGatherGetCollByteCount, defaultInitColl, AllGatherInitData,
                                 AllGatherGetBw, AllGatherRunColl};

void AllGatherGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AllGatherRunTest(struct testArgs* args)
{
  args->collTest = &allGatherTest;
  mscclppDevConn_t* devConns;
  int nCons;
  MSCCLPPCHECK(mscclppGetAllDeviceConnections(args->comm, &devConns, &nCons));
  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * nCons));
  TESTCHECK(TimeTest(args));
  return testSuccess;
}

struct testEngine allGatherEngine = {AllGatherGetBuffSize, AllGatherRunTest, nullptr, nullptr};

#pragma weak mscclppTestEngine = allGatherEngine
