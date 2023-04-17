#include <cuda/barrier>
#include <tuple>
#include <vector>

#include "comm.h"
#include "common.h"

#define ALIGN 4

const int phase2Tag = 2;
mscclppDevConn_t* conns;
void* scratch = nullptr;
void* sendRecvData = nullptr;
cuda::barrier<cuda::thread_scope_device>* barrier = nullptr;

struct Chunk
{
  size_t offset;
  size_t size;
};

inline int getSendTag(int rank, int peer)
{
  return rank < peer ? 0 : 1;
}

inline int getRecvTag(int rank, int peer)
{
  return rank < peer ? 1 : 0;
}

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx, size_t chunkCount)
{
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t numSmallChunks = chunkCount - numLargeChunks;
  size_t offset =
    (remainder - numLargeChunks) * largeChunkSize + (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, numLargeChunks * largeChunkSize + numSmallChunks * smallChunkSize};
}

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
    conn.putWithSignalAndFlush(dstOffset, srcOffset, size);
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

__device__ void reduceSum(int* dst, int* src, size_t size)
{
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__global__ void initData(int* data, size_t size, int rank)
{
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    data[i] = rank;
  }
}

__global__ void allReduceKernel0(int rank, int nRanks, size_t dataCount, size_t scratchDataCount,
                                 mscclppDevConn_t* conns, void* scratch, void* sendRecvData,
                                 cuda::barrier<cuda::thread_scope_device>* barrier)
{
  int idx = blockIdx.x;
  int peer = peerRank(idx, rank);
  mscclppDevConn_t phase1SendConn = conns[phase1SendConnIdx(peer, rank)];
  mscclppDevConn_t phase1RecvConn = conns[phase1RecvConnIdx(peer, rank)];
  mscclppDevConn_t phase2Conn = conns[phase2ConnIdx(peer, rank)];

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  Chunk toPeerChunk = getChunk(dataCount, nRanks, peer, 1);
  // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
  // The destination will have allocated a scratch buffer of size numPeers() * toPeerChunk.size and
  // inside that each of the destination's peers send to the nth chunk, where n is the index of the
  // source peer from the destination's perspective.
  size_t dstOffset = peerIdx(rank, peer) * toPeerChunk.size;
  send(phase1SendConn, toPeerChunk.offset * sizeof(int), dstOffset * sizeof(int), toPeerChunk.size * sizeof(int));
  recv(phase1RecvConn);

  if (threadIdx.x == 0)
    barrier->arrive_and_wait();
  __syncthreads();

  // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
  Chunk rankChunk = getChunk(dataCount, nRanks, rank, 1);
  int* chunk = (int*)sendRecvData + rankChunk.offset;
  int numPeers = nRanks - 1, numBlocks = nRanks - 1;
  Chunk blockUserChunk = getChunk(rankChunk.size, numBlocks, idx, 1);
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    assert(scratchDataCount % numPeers == 0);
    assert(scratchDataCount / numPeers == rankChunk.size);
    size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
    int* scratchChunk = (int*)scratch + peerIdx * scratchDataCountPerPeer;
    Chunk blockScratchChunk = getChunk(scratchDataCountPerPeer, numBlocks, idx, 1);
    assert(blockScratchChunk.size == blockUserChunk.size);
    reduceSum(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset, blockScratchChunk.size);
  }

  if (threadIdx.x == 0)
    barrier->arrive_and_wait();
  __syncthreads();

  // 2nd communication phase: send the now reduced data between the user buffers
  Chunk collectionChunk = getChunk(dataCount, nRanks, rank, 1);
  send(phase2Conn, collectionChunk.offset * sizeof(int), collectionChunk.offset * sizeof(int),
       collectionChunk.size * sizeof(int));
  recv(phase2Conn);
}

void AllReduceGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                               size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / ALIGN) * ALIGN;
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
  size_t recvcount = args->expectedBytes / sizeof(int);
  // int nranks = args->totalProcs;

  CUDACHECK(cudaSetDevice(args->gpuNum));
  CUDACHECK(cudaMemset(args->recvbuff, 0, args->expectedBytes));
  initData<<<1, 256>>>((int*)args->recvbuff, recvcount, args->proc);

  int* dataHost = new int[recvcount];
  for (size_t i = 0; i < recvcount; i++) {
    dataHost[i] = args->totalProcs * (args->totalProcs - 1) / 2;
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
  double factor = (2 * (double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t AllReduceRunColl(void* sendbuff, void* recvbuff, int nranksPerNode, size_t nBytes, mscclppComm_t comm,
                              cudaStream_t stream, int kernel_num)
{
  int worldSize = comm->nRanks;
  int nPeers = worldSize - 1;
  int dataCount = nBytes / sizeof(int);
  Chunk chunk = getChunk(dataCount, worldSize, comm->rank, 1);
  size_t scratchDataCount = chunk.size * nPeers;
  allReduceKernel0<<<worldSize - 1, 256, 0, stream>>>(comm->rank, worldSize, dataCount, scratchDataCount, conns,
                                                      scratch, sendRecvData, barrier);
  return testSuccess;
}

struct testColl allReduceTest = {"AllReduce", AllReduceGetCollByteCount, AllReduceInitData, AllReduceGetBw,
                                 AllReduceRunColl};

testResult_t AllReduceSetupMscclppConnections(struct testArgs* args)
{
  int rank = args->proc, worldSize = args->totalProcs;
  size_t bufferSize = args->maxbytes;
  Chunk chunk = getChunk(bufferSize / sizeof(int), args->totalProcs, rank, 1);
  int nPeers = args->totalProcs - 1;
  size_t scratchBytes = chunk.size * nPeers * sizeof(int);

  CUDACHECK(cudaMalloc(&scratch, scratchBytes));

  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer != args->proc) {
      int sendTag = getSendTag(args->proc, peer);
      int recvTag = getRecvTag(args->proc, peer);
      MSCCLPPCHECK(mscclppConnect(args->comm, peer, sendTag, args->recvbuff, bufferSize, mscclppTransportP2P, nullptr));
      MSCCLPPCHECK(mscclppConnect(args->comm, peer, recvTag, scratch, scratchBytes, mscclppTransportP2P, nullptr));
      MSCCLPPCHECK(
        mscclppConnect(args->comm, peer, phase2Tag, args->recvbuff, bufferSize, mscclppTransportP2P, nullptr));
    }
  }
  MSCCLPPCHECK(mscclppConnectionSetup(args->comm));

  return testSuccess;
}

testResult_t AllReduceTeardownMscclppConnections()
{
  if (scratch != nullptr) {
    CUDACHECK(cudaFree(scratch));
    scratch = nullptr;
  }
  return testSuccess;
}

testResult_t AllReduceRunTest(struct testArgs* args)
{
  args->collTest = &allReduceTest;

  sendRecvData = args->recvbuff;
  CUDACHECK(cudaMalloc(&barrier, sizeof(cuda::barrier<cuda::thread_scope_device>)));
  cuda::barrier<cuda::thread_scope_device> initBarrier(args->totalProcs - 1);
  CUDACHECK(
    cudaMemcpy(barrier, &initBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
  int nPeers = args->totalProcs - 1;
  int rank = args->proc;
  std::vector<mscclppDevConn_t> hostConns(nPeers * 3, mscclppDevConn_t());

  for (int peer = 0; peer < args->totalProcs; ++peer) {
    mscclppDevConn_t* devConn;
    if (peer != rank) {
      int sendTag = getSendTag(args->proc, peer);
      int recvTag = getRecvTag(args->proc, peer);
      MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, peer, sendTag, &devConn));
      hostConns[phase1SendConnIdx(peer, rank)] = *devConn;
      MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, peer, recvTag, &devConn));
      hostConns[phase1RecvConnIdx(peer, rank)] = *devConn;
      MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, peer, phase2Tag, &devConn));
      hostConns[phase2ConnIdx(peer, rank)] = *devConn;
    }
  }
  CUDACHECK(cudaMalloc(&conns, nPeers * 3 * sizeof(mscclppDevConn_t)));
  CUDACHECK(cudaMemcpy(conns, hostConns.data(), hostConns.size() * sizeof(mscclppDevConn_t), cudaMemcpyHostToDevice));

  TESTCHECK(TimeTest(args));

  CUDACHECK(cudaFree(barrier));
  CUDACHECK(cudaFree(conns));

  return testSuccess;
}

struct testEngine allReduceEngine = {AllReduceGetBuffSize, AllReduceRunTest, AllReduceSetupMscclppConnections,
                                     AllReduceTeardownMscclppConnections};

#pragma weak mscclppTestEngine = allReduceEngine
