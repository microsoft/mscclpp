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

// Measure current time in second.
static double getTime(void)
{
  struct timespec tspec;
  if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
    printf("clock_gettime failed\n");
    exit(EXIT_FAILURE);
  }
  return (tspec.tv_nsec / 1.0e9) + tspec.tv_sec;
}

__constant__ mscclppDevConn_t constDevConns[16];

__global__ void kernel(int rank, int world_size)
{
  if (threadIdx.x % 32 != 0) return;

  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  mscclppDevConn_t devConn = constDevConns[remoteRank];
  volatile int *data = (volatile int *)devConn.localBuff;
  volatile uint64_t *localFlag = devConn.localFlag;
#if (USE_DMA_FOR_P2P == 0)
  volatile uint64_t *remoteFlag = devConn.remoteFlag;
#endif
  volatile uint64_t *proxyFlag = devConn.proxyFlag;

  uint64_t baseFlag = *localFlag;

  if (threadIdx.x == 0) {
    // Set my data and flag
    *(data + rank) = rank + 1;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Do we need a sys fence?
    // __threadfence_system();
    *localFlag = baseFlag + 1;
  }

  // get a thread-local trigger and a request for waiting on it
  mscclppTrigger_t trig;
  mscclppRequest_t req = devConn.fifo.getTrigger(&trig);

  // Each warp receives data from different ranks
#if (USE_DMA_FOR_P2P == 1)

  // Trigger sending data, flag and synchronize after
  devConn.fifo.setTrigger(trig, mscclppFlag | mscclppData | mscclppSync, rank * sizeof(int), sizeof(int));

  // Wait on the request to make sure it is safe to reuse buffer and flag
  devConn.fifo.waitTrigger(req);

  // Wait for receiving data from remote rank
  while (*proxyFlag == baseFlag) {}

#else // USE_DMA_FOR_P2P == 0

  if (devConn.remoteBuff == NULL) { // IB
    // Wait until the proxy have sent my data and flag
    devConn.waitTrigger(trig);

    // Trigger sending data and flag
    devConn.setTrigger(trig, mscclppFlag | mscclppData, rank * sizeof(int), sizeof(int));

    // Wait for receiving data from remote rank
    while (*proxyFlag == baseFlag) {}
  } else { // P2P
    // Directly read data
    volatile int *remoteData = (volatile int *)devConn.remoteBuff;

    // Wait until the remote data is set
    while (*remoteFlag == baseFlag) {}

    // Read remote data
    data[remoteRank] = remoteData[remoteRank];
  }

#endif
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
  size_t data_size = sizeof(int) * world_size;
  CUDACHECK(cudaMalloc(&data_d, data_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(uint64_t)));
  CUDACHECK(cudaMemset(data_d, 0, data_size));
  CUDACHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

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

  kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size);
  CUDACHECK(cudaDeviceSynchronize());

  // Read results from GPU
  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  CUDACHECK(cudaMemcpy(buf, data_d, sizeof(int) * world_size, cudaMemcpyDeviceToHost));

  bool failed = false;
  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i + 1) {
      printf("rank: %d, wrong data: %d, expected %d\n", rank, buf[i], i + 1);
      failed = true;
    }
  }
  if (failed) {
    return -1;
  }

  // Perf test
  cudaEvent_t ev_start;
  cudaEvent_t ev_end;
  CUDACHECK(cudaEventCreate(&ev_start));
  CUDACHECK(cudaEventCreate(&ev_end));

  // warm up
  // int warmupiter = 10;
//  for (int i = 0; i < warmupiter; ++i) {
//    kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size);
//  }

  // cudaGraph Capture
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  int cudagraphiter = 100;
  for (int i = 0; i < cudagraphiter; ++i) {
  	kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size);
  }
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

  int cudagraphwarmup = 10;
  for (int i = 0; i < cudagraphwarmup; ++i) {
	  cudaGraphLaunch(instance, stream);
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  // measure runtime 
//  CUDACHECK(cudaEventRecord(ev_start, stream));
  double t0 = getTime();
  int cudagraphlaunch = 10;
  for (int i = 0; i < cudagraphlaunch; ++i) {
  // kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size);
     cudaGraphLaunch(instance, stream);
  }
//  CUDACHECK(cudaEventRecord(ev_end, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  double t1 = getTime();
  float ms = (t1-t0)*1000.0;
//  CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
  printf("rank: %d, time: %f us/iter\n", rank, ms * 1000. / (float) cudagraphlaunch / (float) cudagraphiter);

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
