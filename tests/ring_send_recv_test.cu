#include "mscclpp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "common.h"

#define MSCCLPP_USE_MPI_FOR_TESTS
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include <mpi.h>
#endif // MSCCLPP_USE_MPI_FOR_TESTS

#define RANKS_PER_NODE 8
#define USE_DMA_FOR_P2P 1
#define TEST_CONN_TYPE 0 // 0: P2P(for local)+IB(for remote), 1: IB-Only

#define MSCCLPPCHECK(call)                                                                                             \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      /* Print the back trace*/                                                                                        \
      printf("Failure at %s:%d -> %d\n", __FILE__, __LINE__, res);                                                     \
      return res;                                                                                                      \
    }                                                                                                                  \
  } while (0);

// Check CUDA RT calls
#define CUDACHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      printf("%s:%d Cuda failure '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (false)

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


void parse_arguments(int argc, const char* argv[], const char** ip_port, int* rank, int* world_size)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  if (argc == 4) {
    *rank = atoi(argv[2]);
    *world_size = atoi(argv[3]);
  } else {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
  }
#else
  if (argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  *rank = atoi(argv[2]);
  *world_size = atoi(argv[3]);
#endif
}

__global__ void initKernel(char* data_d, int dataSize)
{
  for (size_t i = threadIdx.x; i < dataSize; i += blockDim.x) {
    data_d[i] = i % 256;
  }
}

__constant__ mscclppDevConn_t sendConnConst;
__constant__ mscclppDevConn_t recvConnConst;

__global__ void smKernel(bool root, size_t dataSize)
{
  mscclppDevConn_t sendConn = sendConnConst;
  mscclppDevConn_t recvConn = recvConnConst;

  if (root)
  {
    sendConn.putDirect(0, dataSize, threadIdx.x, blockDim.x);
    sendConn.signalDirect(threadIdx.x, blockDim.x);
    recvConn.wait();
  }
  else
  {
    recvConn.wait();
    sendConn.putDirect(0, dataSize, threadIdx.x, blockDim.x);
    sendConn.signalDirect(threadIdx.x, blockDim.x);
  }
}

__global__ void dmaKernel(bool root, size_t dataSize)
{
  if (threadIdx.x != 0)
    return;

  mscclppDevConn_t sendConn = sendConnConst;
  mscclppDevConn_t recvConn = recvConnConst;

  if (root)
  {
    sendConn.putWithSignal(0, dataSize);
    recvConn.wait();
  }
  else
  {
    recvConn.wait();
    sendConn.putWithSignal(0, dataSize);
  }
}

void resetData(char* data_d, size_t data_size, bool isRoot)
{
  if (isRoot) {
    initKernel<<<1, 256>>>(data_d, data_size);
  } else {
    CUDACHECK(cudaMemset(data_d, 0, data_size));
  }
}

int main(int argc, const char* argv[])
{
  const char* ip_port;
  int rank, world_size;
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
  parse_arguments(argc, argv, &ip_port, &rank, &world_size);

  bool isRoot = rank == 0;

  CUDACHECK(cudaSetDevice(rank));

  if (rank == 0)
    printf("Initializing MSCCL++\n");
  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, ip_port, rank));

  char* data_d;
  // size_t data_size = 1 << 10; // Kilobyte
  // size_t data_size = 1 << 20; // Megabyte
  size_t data_size = 1 << 30; // Gigabyte
  CUDACHECK(cudaMalloc(&data_d, data_size));
  resetData(data_d, data_size, isRoot);

  if (rank == 0) {
    MSCCLPPCHECK(mscclppConnect(comm, 1, 0, data_d, data_size, mscclppTransportP2P));
  } else {
    MSCCLPPCHECK(mscclppConnect(comm, 0, 0, data_d, data_size, mscclppTransportP2P));
  }
  if (rank == 0)
    printf("Finished connection\n");

  MSCCLPPCHECK(mscclppConnectionSetup(comm));
  if (rank == 0)
    printf("Finished Setup\n");

  MSCCLPPCHECK(mscclppProxyLaunch(comm));
  if (rank == 0)
    printf("Finished proxy launch\n");

  mscclppDevConn_t *sendDevConn;
  mscclppDevConn_t *recvDevConn;
  if (rank == 0) {
    MSCCLPPCHECK(mscclppGetDeviceConnection(comm, 1, 0, &sendDevConn));
    MSCCLPPCHECK(mscclppGetDeviceConnection(comm, 1, 0, &recvDevConn));
  } else {
    MSCCLPPCHECK(mscclppGetDeviceConnection(comm, 0, 0, &sendDevConn));
    MSCCLPPCHECK(mscclppGetDeviceConnection(comm, 0, 0, &recvDevConn));
  }
  if (rank == 0)
    printf("Finished device connection\n");

  CUDACHECK(cudaMemcpyToSymbol(sendConnConst, sendDevConn, sizeof(mscclppDevConn_t)));
  CUDACHECK(cudaMemcpyToSymbol(recvConnConst, recvDevConn, sizeof(mscclppDevConn_t)));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  if (rank == 0)
    printf("Start running kernel\n");
  smKernel<<<1, 256, 0, stream>>>(isRoot, data_size);
  CUDACHECK(cudaDeviceSynchronize());

  // Read results from GPU
  char* buf = (char*)calloc(data_size, 1);
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  CUDACHECK(cudaMemcpy(buf, data_d, data_size, cudaMemcpyDeviceToHost));

  bool failed = false;
  for (size_t i = 0; i < data_size; ++i) {
    char expected = (char)(i % 256);
    if (buf[i] != expected) {
      printf("rank: %d, wrong data: %d, expected %d\n", rank, buf[i], expected);
      failed = true;
    }
  }
  if (failed) {
    return -1;
  }

/*
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
  float ms = (t1 - t0) * 1000.0;
  //  CUDACHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
  printf("rank: %d, time: %f us/iter\n", rank, ms * 1000. / (float)cudagraphlaunch / (float)cudagraphiter);

*/

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
