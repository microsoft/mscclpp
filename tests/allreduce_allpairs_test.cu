#include <cuda/barrier>
#include <tuple>
#include <vector>

#include "comm.h"
#include "common.h"

#define ALIGN 4

namespace {
__global__ std::vector<mscclppDevConn_t> devConns;
void* scratch = nullptr;
void* userData = nullptr;
cuda::barrier<cuda::thread_scope_device>* barrier = nullptr;

typedef void reduceFunc(int* dst, int* src, size_t size);

struct Volume
{
  size_t offset;
  size_t size;
};

__host__ __device__ Volume chunkVolume(size_t dataCount, size_t numChunks, size_t chunkIdx, size_t chunkCount)
{
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t numSmallChunks = chunkCount - numLargeChunks;
  size_t offset =
    (remainder - numLargeChunks) * largeChunkSize + (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Volume{offset, numLargeChunks * largeChunkSize + numSmallChunks * smallChunkSize};
}
} // namespace

__host__ __device__ int peerIdx(int peerRank, int rank)
{
    return peerRank < rank ? peerRank : peerRank - 1;
}

__host__ __device__ int peerRank(int peerIdx, int rank)
{
    return peerIdx < rank ? peerIdx : peerIdx + 1;
}

__host__ __device__ int phase1SendConnIdx(int peerRank, int rank)
{
    return peerIdx(peerRank, rank) * 3;
}

__host__ __device__ int phase1RecvConnIdx(int peerRank, int rank)
{
    return peerIdx(peerRank, rank) * 3 + 1;
}

__host__ __device__ int phase2ConnIdx(int peerRank, int rank)
{
    return peerIdx(peerRank, rank) * 3 + 2;
}

__device__ void send(mscclppDevConn_t& conn, size_t srcOffset, size_t dstOffset, size_t size)
{
    if (threadIdx.x == 0) {
      conn.putWithSignalAndFlush(srcOffset, dstOffset, size);
    }
    __syncthreads();
}

__device__ void recv(mscclppDevConn_t& conn)
{
    if (threadIdx.x == 0) {
      conn.wait();
    }
    __syncthreads();
}

template <typename T> __device__ void reduceSum(T* dst, T* src, size_t size)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      dst[i] += src[i];
    }
}

template <typename T> __global__ void initData(T* data, size_t size, int rank)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      data[i] = rank;
    }
}

__device__ void allReduceKernel0(int rank, int nRanks, int dataCount, int scratchDataCount, reduceFunc reduce)
{
    int idx = blockIdx.x;
    int peer = peerRank(idx, rank);
    mscclppDevConn_t phase1SendConn = devConns[phase1SendConnIdx(peer, rank)];
    mscclppDevConn_t phase1RecvConn = devConns[phase1RecvConnIdx(peer, rank)];
    mscclppDevConn_t phase2Conn = devConns[phase2ConnIdx(peer, rank)];

    // 1st communication phase: send data to the scratch buffer of the peer associated with this block
    Volume toPeer = chunkVolume(dataCount, nRanks, peer, 1);
    // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
    // The destination will have allocated a scratch buffer of size numPeers() * toPeer.size and
    // inside that each of the destination's peers send to the nth chunk, where n is the index of the
    // source peer from the destination's perspective.
    size_t dstOffset = peerIdx(rank, peer) * toPeer.size;
    send(phase1SendConn, toPeer.offset * sizeof(int), dstOffset * sizeof(int), toPeer.size * sizeof(int));
    recv(phase1RecvConn);

    if (threadIdx.x == 0)
      barrier->arrive_and_wait();
    __syncthreads();

    // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
    Volume rankChunk = chunkVolume(dataCount, nRanks, rank, 1);
    int* chunk = (int*)userData + rankChunk.offset;
    int numPeers = nRanks - 1, numBlocks = nRanks - 1;
    Volume blockUserChunk = chunkVolume(rankChunk.size, numBlocks, idx, 1);
    for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
      assert(scratchDataCount % numPeers == 0);
      assert(scratchDataCount / numPeers == rankChunk.size);
      size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
      int* scratchChunk = (int*)scratch + peerIdx * scratchDataCountPerPeer;
      Volume blockScratchChunk = chunkVolume(scratchDataCountPerPeer, numBlocks, idx, 1);
      assert(blockScratchChunk.size == blockUserChunk.size);
      reduce(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset, blockScratchChunk.size);
    }

    if (threadIdx.x == 0)
      barrier->arrive_and_wait();
    __syncthreads();

    // 2nd communication phase: send the now reduced data between the user buffers
    Volume srcVolume2 = chunkVolume(dataCount, nRanks, rank, 1);
    send(phase2Conn, srcVolume2.offset, srcVolume2.offset, srcVolume2.size);
    recv(phase2Conn);
}

void AllReduceGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                               size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / (ALIGN * nranks)) * ALIGN;
  *sendcount = base;
  *recvcount = base;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

void AllReduceGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AllReduceInitData(struct testArgs* args, int in_place)
{
  size_t sendcount = args->sendBytes / sizeof(int);
  size_t recvcount = args->expectedBytes / sizeof(int);
  // int nranks = args->totalProcs;

  CUDACHECK(cudaSetDevice(args->gpuNum));
  int rank = args->proc;
  CUDACHECK(cudaMemset(args->recvbuff, 0, args->expectedBytes));
  initData<<<1, 256>>>((int*)args->recvbuff, recvcount, args->proc);

  int* dataHost = new int[recvcount];
  for (size_t i = 0; i < recvcount; i++) {
    dataHost[i] =  args->totalProcs * (args->totalProcs - 1) / 2;
  }
  CUDACHECK(cudaMemcpy(args->expected, dataHost, recvcount * sizeof(int), cudaMemcpyHostToDevice));
  delete dataHost;
  CUDACHECK(cudaDeviceSynchronize());
  MSCCLPPCHECK(mscclppBootstrapBarrier(args->comm));
  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks)
{
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}


testResult_t AllReduceRunColl(void* sendbuff, void* recvbuff, int nranksPerNode, size_t nBytes, mscclppComm_t comm,
                              cudaStream_t stream, int kernel_num)
{
  int worldSize = comm->nRanks;
  allReduceKernel0<<<worldSize - 1, 256>>>(comm->rank, worldSize, nBytes / sizeof(int), nBytes / sizeof(int),
                                           reduceSum<int>);
  return testSuccess;
}

struct testColl allReduceTest = {"AllReduce", AllReduceGetCollByteCount, AllReduceInitData, AllReduceGetBw,
                                 AllReduceRunColl};

testResult_t AllReduceSetupMscclppConnections(struct testArgs* args)
{
  Volume chunk = chunkVolume(args->nbytes, args->totalProcs, args->proc, 1);
  int nRanks = args->totalProcs - 1;
  size_t scratchBytes = chunk.size * nRanks * sizeof(int);

  CUDACHECK(cudaMalloc(&scratch, scratchBytes));

  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer != args->proc) {
      int sendTag = args->proc < peer ? 0 : 1;
      int recvTag = args->proc < peer ? 1 : 0;
      MSCCLPPCHECK(
        mscclppConnect(args->comm, peer, sendTag, args->recvbuff, args->expectedBytes, mscclppTransportP2P, nullptr));
      MSCCLPPCHECK(
        mscclppConnect(args->comm, peer, recvTag, scratch, scratchBytes, mscclppTransportP2P, nullptr));
      MSCCLPPCHECK(
        mscclppConnect(args->comm, peer, 2, args->recvbuff, args->expectedBytes, mscclppTransportP2P, nullptr));
    }
  }

  return testSuccess;
}

testResult_t AllReduceRunTest(struct testArgs* args)
{
  args->collTest = &allReduceTest;
  mscclppDevConn_t* devConns;
  int nCons;
  MSCCLPPCHECK(mscclppGetAllDeviceConnections(args->comm, &devConns, &nCons));
  CUDACHECK(cudaMalloc(&barrier, sizeof(cuda::barrier<cuda::thread_scope_device>)));
  cuda::barrier<cuda::thread_scope_device> initBarrier(args->totalProcs - 1);
  CUDACHECK(
    cudaMemcpy(barrier, &initBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
  // CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * nCons));
  TESTCHECK(TimeTest(args));
  CUDACHECK(cudaFree(barrier));
  return testSuccess;
}

struct testEngine allReduceEngine = {AllReduceGetBuffSize, AllReduceRunTest, AllReduceSetupMscclppConnections};

#pragma weak mscclppTestEngine = allReduceEngine