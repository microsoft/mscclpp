#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

#define RANKS_PER_NODE 8
#define USE_DMA_FOR_P2P 1
#define TEST_CONN_TYPE 0 // 0: P2P(for local)+IB(for remote), 1: IB-Only

#define NELEM 256 // only up to 256 for now.
#define NELEM2 (NELEM / 2)
#define NUM_THREADS_PER_REMOTE_RANK 128

#define WAIT_LOOP(cond) do { \
  constexpr int maxIter = 10000000; \
  int iter = 0; \
  while (cond) { \
    if (iter++ == maxIter) { \
      printf("Potential infinite loop detected at %s:%d (" #cond ")\n", __FILE__, __LINE__); \
    } \
  } \
} while (0);

#define MSCCLPPCHECK(call) do { \
  mscclppResult_t res = call; \
  if (res != mscclppSuccess && res != mscclppInProgress) { \
    /* Print the back trace*/ \
    printf("Failure at %s:%d -> %d\n", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if( err != cudaSuccess ) {                                \
        printf("%s:%d Cuda failure '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(false)

__constant__ mscclppDevConn_t constDevConns[16];

__global__ void kernel(int rank, int world_size, uint64_t flag_base, uint64_t iter)
{
  int thDiv = threadIdx.x / NUM_THREADS_PER_REMOTE_RANK;
  int thMod = threadIdx.x % NUM_THREADS_PER_REMOTE_RANK;
  int remoteRank = (thDiv < rank) ? thDiv : thDiv + 1;
  mscclppDevConn_t devConn = constDevConns[remoteRank];
  volatile int *data = (volatile int *)devConn.localBuff;
  volatile uint64_t *localFlag = devConn.localFlag;
  volatile uint64_t *remoteFlag = devConn.remoteFlag;
  volatile uint64_t *proxyFlag = devConn.proxyFlag;
  volatile uint64_t *trig = (volatile uint64_t *)devConn.trigger;

  for (uint64_t i = flag_base; i < iter + flag_base; i++) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      *localFlag = i + 1;
    }

    // Each warp receives data from different ranks
#if (USE_DMA_FOR_P2P == 1)

    // Trigger sending data and flag
    if (thMod == 0) {
      uint64_t dataOffset = rank * sizeof(int) * NELEM;
      uint64_t dataSize = sizeof(int) * NELEM;
      *trig = TRIGGER_VALUE(mscclppSync | mscclppFlag | mscclppData, dataOffset, dataSize);

      // Wait until the proxy have sent my data and flag
      WAIT_LOOP(*trig != 0);

      // Wait for receiving data from remote rank
      WAIT_LOOP(*proxyFlag == i);
    }
    __syncthreads();

#else // USE_DMA_FOR_P2P == 0

    if (devConn.remoteBuff == NULL) { // IB
      if (thMod == 0) {
        // Trigger sending data and flag
        uint64_t dataOffset = rank * sizeof(int) * NELEM;
        uint64_t dataSize = sizeof(int) * NELEM;
        *trig = TRIGGER_VALUE(mscclppSync | mscclppFlag | mscclppData, dataOffset, dataSize);

        // Wait until the proxy have sent my data and flag
        while (*trig != 0) {}

        // Wait for receiving data from remote rank
        while (*proxyFlag == i) {}
      }
    } else { // P2P
      // Wait until the remote data is set
        while (*remoteFlag == i) {}

      // Read remote data
      volatile uint64_t *pDst = (volatile uint64_t *)devConn.localBuff;
      volatile uint64_t *pSrc = (volatile uint64_t *)devConn.remoteBuff;
      pDst[NELEM2 * remoteRank + thMod] = pSrc[NELEM2 * remoteRank + thMod];
      __syncthreads();
    }
#endif
  }
}

int rankToLocalRank(int rank)
{
  return rank % RANKS_PER_NODE;
}

int rankToNode(int rank)
{
  return rank / RANKS_PER_NODE;
}

int cudaNumToIbNum(int cudaNum)
{
  int ibNum;
  if (cudaNum == 0) {
    ibNum = 0;
  } else if (cudaNum == 1) {
    ibNum = 4;
  } else if (cudaNum == 2) {
    ibNum = 1;
  } else if (cudaNum == 3) {
    ibNum = 5;
  } else if (cudaNum == 4) {
    ibNum = 2;
  } else if (cudaNum == 5) {
    ibNum = 6;
  } else if (cudaNum == 6) {
    ibNum = 3;
  } else if (cudaNum == 7) {
    ibNum = 7;
  } else {
    printf("Invalid cudaNum: %d\n", cudaNum);
    exit(EXIT_FAILURE);
  }
  return ibNum;
}

void print_usage(const char *prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

int main(int argc, const char *argv[])
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  int rank;
  int world_size;
  if (argc == 4) {
    rank = atoi(argv[2]);
    world_size = atoi(argv[3]);
  } else {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  }
#else
  if (argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  int rank = atoi(argv[2]);
  int world_size = atoi(argv[3]);
#endif
  int localRank = rankToLocalRank(rank);
  int thisNode = rankToNode(rank);
  int cudaNum = localRank;
  int ibNum = cudaNumToIbNum(cudaNum);

  CUDACHECK(cudaSetDevice(cudaNum));
  std::string ibDevStr = "mlx5_ib" + std::to_string(ibNum);

  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));

  int *data_d;
  uint64_t *flag_d;
  size_t data_size = sizeof(int) * NELEM * world_size;
  CUDACHECK(cudaMalloc(&data_d, data_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(uint64_t)));
  CUDACHECK(cudaMemset(data_d, 0, data_size));
  CUDACHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

  int *in_buf = (int *)calloc(NELEM * world_size, sizeof(int));
  if (in_buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  for (int i = NELEM * rank; i < NELEM * (rank + 1); ++i) {
    in_buf[i] = rank + 1;
  }
  CUDACHECK(cudaMemcpy(data_d, in_buf, data_size, cudaMemcpyHostToDevice));

  mscclppDevConn_t devConns[16];
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    mscclppTransport_t transportType = mscclppTransportIB;
    const char *ibDev = ibDevStr.c_str();
#if (TEST_CONN_TYPE == 0) // P2P+IB
    if (rankToNode(r) == thisNode) {
      transportType = mscclppTransportP2P;
      ibDev = NULL;
    }
#endif
    // Connect with all other ranks
    MSCCLPPCHECK(mscclppConnect(comm, &devConns[r], r, data_d, data_size, flag_d, 0, transportType, ibDev));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));

  MSCCLPPCHECK(mscclppProxyLaunch(comm));

  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * world_size));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaDeviceSynchronize());

  kernel<<<1, NUM_THREADS_PER_REMOTE_RANK * (world_size - 1), 0, stream>>>(rank, world_size, 0, 1);
  CUDACHECK(cudaStreamSynchronize(stream));

  // Read results from GPU
  int *buf = (int *)calloc(NELEM * world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  CUDACHECK(cudaMemcpy(buf, data_d, data_size, cudaMemcpyDeviceToHost));

  bool failed = false;
  for (int i = 0; i < NELEM * world_size; ++i) {
    if (buf[i] != (i / NELEM) + 1) {
      printf("rank: %d, wrong data: %d, expected %d, index %d\n", rank, buf[i], (i / NELEM) + 1, i);
      i += NELEM - 1;
      failed = true;
    }
  }
  if (failed) {
    return -1;
  }

  // measure runtime
  double t0 = MPI_Wtime();
  int iter = 100;
  kernel<<<1, NUM_THREADS_PER_REMOTE_RANK * (world_size - 1), 0, stream>>>(rank, world_size, 1, iter);
  CUDACHECK(cudaStreamSynchronize(stream));

  double t1 = MPI_Wtime();
  float ms = (t1-t0)*1000.0;
  printf("rank: %d, time: %f us/iter\n", rank, ms * 1000. / (float)iter);

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
