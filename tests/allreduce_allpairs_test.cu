#include "mscclpp.h"
#include <tuple>
#include <vector>
#include <cuda/barrier>

#include "common.h"

#define MSCCLPPCHECK(call) do { \
  mscclppResult_t res = call; \
  if (res != mscclppSuccess && res != mscclppInProgress) { \
  /* Print the back trace*/ \
  printf("Failure at %s:%d -> %d\n", __FILE__, __LINE__, res);    \
  return res; \
  } \
} while (0);

#define CUDACHECK(cmd) do { \
  cudaError_t err = cmd; \
  if( err != cudaSuccess ) { \
    printf("%s:%d Cuda failure '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while(false)

struct Volume {
  size_t offset;
  size_t size;
};

__host__ __device__ Volume chunkVolume(size_t totalSize, size_t totalChunks, size_t chunkIdx, size_t chunkCount) {
  size_t remainder = totalSize % totalChunks;
  size_t smallChunk = totalSize / totalChunks;
  size_t largeChunk = smallChunk + 1;
  size_t numLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t numSmallChunks = chunkCount - numLargeChunks;
  size_t offset = (remainder - numLargeChunks) * largeChunk +
                  (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunk;
  return Volume{offset, numLargeChunks * largeChunk + numSmallChunks * smallChunk};
}

template<class T, void (*reduce)(T*,T*,size_t)>
struct AllreduceAllpairs {
  int rank;
  int nRanks;
  T* userData;
  size_t userSize;
  T* scratch;
  size_t scratchSize;
  mscclppDevConn_t* conns;
  uint64_t* connFlags;
  cuda::barrier<cuda::thread_scope_device>* barrier;

  __device__ void run(int idx) {
    int myPeer = peerRank(idx, rank);
    mscclppDevConn_t phase1SendConn = conns[phase1SendConnIdx(myPeer)];
    mscclppDevConn_t phase1RecvConn = conns[phase1RecvConnIdx(myPeer)];
    mscclppDevConn_t phase2Conn = conns[phase2ConnIdx(myPeer)];

    // 1st communication phase: send data to the scratch buffer of the peer associated with this block
    Volume toPeer = chunkVolume(userSize, nRanks, myPeer, 1);
    // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
    // The destination will have allocated a scratch buffer of size numPeers() * toPeer.size and
    // inside that each of the destination's peers send to the nth chunk, where n is the index of the
    // source peer from the destination's perspective.
    size_t dstOffset = peerIdx(rank, myPeer) * toPeer.size;
    send(phase1SendConn, toPeer.offset, dstOffset, toPeer.size);
    recv(phase1RecvConn);

    if (threadIdx.x == 0)
      barrier->arrive_and_wait();
    __syncthreads();

    // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
    Volume rankUserChunk = chunkVolume(userSize, nRanks, rank, 1);
    T* userChunk = userData + rankUserChunk.offset;
    Volume blockUserChunk = chunkVolume(rankUserChunk.size, numBlocks(), idx, 1);
    for (int peerIdx = 0; peerIdx < numPeers(); ++peerIdx) {
      assert(scratchSize % numPeers() == 0);
      assert(scratchSize / numPeers() == rankUserChunk.size);
      size_t scratchChunkSize = scratchSize / numPeers();
      T* scratchChunk = scratch + peerIdx * scratchChunkSize;
      Volume blockScratchChunk = chunkVolume(scratchChunkSize, numBlocks(), idx, 1);
      assert(blockScratchChunk.size == blockUserChunk.size);
      reduce(userChunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset, blockScratchChunk.size);
    }

    if (threadIdx.x == 0)
      barrier->arrive_and_wait();
    __syncthreads();

    // 2nd communication phase: send the now reduced data between the user buffers
    Volume srcVolume2 = chunkVolume(userSize, nRanks, rank, 1);
    send(phase2Conn, srcVolume2.offset, srcVolume2.offset, srcVolume2.size);
    recv(phase2Conn);

  }

  __device__ void send(mscclppDevConn_t& conn, size_t srcOffset, size_t dstOffset, size_t size) {
    if (threadIdx.x == 0) {
      volatile uint64_t *localFlag = conn.localFlag;
      *localFlag = 1; // 1 is used to signal the send

      mscclppTrigger_t trigger;
      auto request = conn.fifo.getTrigger(&trigger);
      conn.fifo.setTrigger(trigger, mscclppData | mscclppFlag, srcOffset * sizeof(T), dstOffset * sizeof(T), size * sizeof(T));
    }
    __syncthreads();
  }

  __device__ void recv(mscclppDevConn_t& conn) {
    if (threadIdx.x == 0) {
      volatile uint64_t *proxyFlag = conn.proxyFlag;
      while (*proxyFlag != 1) {}
      *proxyFlag = 0;
    }
    __syncthreads();
  }

  __host__ __device__ int numPeers() {
    return nRanks - 1;
  }

  __host__ __device__ int numBlocks() {
    return numPeers();
  }

  __host__ __device__ int peerIdx(int peerRank, int myRank) {
    return peerRank < myRank ? peerRank : peerRank - 1;
  }

  __host__ __device__ int peerRank(int peerIdx, int myRank) {
    return peerIdx < myRank ? peerIdx : peerIdx + 1;
  }

  __host__ __device__ int phase1SendConnIdx(int peerRank) {
    return peerIdx(peerRank, rank) * 3;
  }

  __host__ __device__ int phase1RecvConnIdx(int peerRank) {
    return peerIdx(peerRank, rank) * 3 + 1;
  }

  __host__ __device__ int phase2ConnIdx(int peerRank) {
    return peerIdx(peerRank, rank) * 3 + 2;
  }

  void freeGPUResources() {
    if (scratch)
      CUDACHECK(cudaFree(scratch));
    scratch = nullptr;
    if (connFlags)
      CUDACHECK(cudaFree(connFlags));
    connFlags = nullptr;
    if (conns)
      CUDACHECK(cudaFree(conns));
    conns = nullptr;
    if (barrier)
      CUDACHECK(cudaFree(barrier));
    barrier = nullptr;
  }
};

// The builder class encapsulates the 
template<class T, void (*reduce)(T*,T*,size_t)>
class AllreduceAllpairsBuilder {
  AllreduceAllpairs<T, reduce> d;
  std::vector<mscclppDevConn_t> hostConns;

public:

  // The constructor is called after the user has allocated the buffer to be allreduced
  AllreduceAllpairsBuilder(T* data, size_t size) {
    d.userData = data;
    d.userSize = size;
    d.scratch = nullptr;
    d.connFlags = nullptr;
    d.conns = nullptr;
    d.barrier = nullptr;
  }

  // connect is called after rank initialization but before connection setup
  mscclppResult_t connect(mscclppComm_t comm) {
    MSCCLPPCHECK(mscclppCommRank(comm, &d.rank));
    MSCCLPPCHECK(mscclppCommSize(comm, &d.nRanks));

    Volume myChunks = chunkVolume(d.userSize, d.nRanks, d.rank, 1);
    d.scratchSize = myChunks.size * d.numPeers();

    CUDACHECK(cudaMalloc(&d.scratch, d.scratchSize * sizeof(T)));
    CUDACHECK(cudaMalloc(&d.connFlags, 3 * sizeof(uint64_t)));
    CUDACHECK(cudaMemset(d.connFlags, 0, 3 * sizeof(uint64_t)));

    hostConns.resize(d.numPeers() * 3);
    for (int peer = 0; peer < d.nRanks; ++peer) {
      if (peer != d.rank) {
        int sendTag = d.rank < peer ? 0 : 1;
        int recvTag = d.rank < peer ? 1 : 0;
        MSCCLPPCHECK(mscclppConnect(comm, hostConns.data() + d.phase1SendConnIdx(peer), peer, d.userData, d.userSize * sizeof(T), d.connFlags + 0, sendTag, mscclppTransportP2P, nullptr));
        MSCCLPPCHECK(mscclppConnect(comm, hostConns.data() + d.phase1RecvConnIdx(peer), peer, d.scratch, d.scratchSize * sizeof(T), d.connFlags + 1, recvTag, mscclppTransportP2P, nullptr));
        MSCCLPPCHECK(mscclppConnect(comm, hostConns.data() + d.phase2ConnIdx(peer), peer, d.userData, d.userSize * sizeof(T), d.connFlags + 2, 2, mscclppTransportP2P, nullptr));
      }
    }

    return mscclppSuccess;
  }

  // finishSetup is called after connection setup and returns an algorithm object that is ready to be passed to a GPU kernel
  AllreduceAllpairs<T, reduce> finishSetup() {
    CUDACHECK(cudaMalloc(&d.conns, hostConns.size() * sizeof(mscclppDevConn_t)));
    CUDACHECK(cudaMemcpy(d.conns, hostConns.data(), hostConns.size() * sizeof(mscclppDevConn_t), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(&d.barrier, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    cuda::barrier<cuda::thread_scope_device> initBarrier(d.numBlocks());
    CUDACHECK(cudaMemcpy(d.barrier, &initBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
    return d;
  }
};

template<class T>
__device__ void reduceSum(T* dst, T* src, size_t size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] += src[i];
  }
}

template<class T>
__global__ void init(T* data, size_t size, int rank) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    data[i] = rank;
  }
}

// The main test kernel
template<class T>
__global__ void testKernel(AllreduceAllpairs<T, reduceSum> d) {
  d.run(blockIdx.x);
}

int main(int argc, const char *argv[]) {
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Init(NULL, NULL);
#endif
  const char* ip_port;
  int rank, world_size;
  parse_arguments(argc, argv, &ip_port, &rank, &world_size);

  CUDACHECK(cudaSetDevice(rank));

  // Allocate and initialize 1 MB of data
  int *data;
  size_t dataSize = 1024 * 1024 / sizeof(int);
  CUDACHECK(cudaMalloc(&data, dataSize * sizeof(int)));
  init<<<1, 256>>>(data, dataSize, rank);
  
  // Create the collective
  AllreduceAllpairsBuilder<int, reduceSum> builder(data, dataSize);
  
  // Create the communicator
  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));

  // Connect the collective
  builder.connect(comm);

  // Finish the setup
  MSCCLPPCHECK(mscclppConnectionSetup(comm));
  MSCCLPPCHECK(mscclppProxyLaunch(comm));
  auto allreduce = builder.finishSetup();

  // Run the collective
  testKernel<<<allreduce.numBlocks(), 256>>>(allreduce);
  
  // Wait for kernel to finish
  CUDACHECK(cudaDeviceSynchronize());

  // Check the result
  int* hostData = new int[dataSize];
  CUDACHECK(cudaMemcpy(hostData, data, dataSize * sizeof(int), cudaMemcpyDeviceToHost));
  int expectedValue = world_size * (world_size - 1) / 2;
  for (size_t i = 0; i < dataSize; ++i) {
    if (hostData[i] != expectedValue) {
      printf("Error at index %lu: %d != %d\n", i, hostData[i], expectedValue);
      return 1;
    }
  }

  MSCCLPPCHECK(mscclppProxyStop(comm));

  MSCCLPPCHECK(mscclppCommDestroy(comm));

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc == 2) {
    MPI_Finalize();
  }
#endif
  printf("Succeeded! %d\n", rank);
  return 0;
}