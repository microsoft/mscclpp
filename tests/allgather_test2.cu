#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define RANKS_PER_NODE 8

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

__constant__ mscclppDevConn_t constDevConns[16];

__global__ void kernel(int rank, int world_size, size_t nelemsPerGPU)
{
  if (threadIdx.x % 32 != 0)
    return;

  int warpId = threadIdx.x / 32;
  bool isIB = false;
  if (warpId >= world_size - 1)
    isIB = true;
  if (isIB)
    warpId = warpId - (world_size - 1);
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  mscclppDevConn_t devConn = constDevConns[remoteRank];
  if (isIB)
    devConn = constDevConns[remoteRank + world_size];

    // Each warp receives data from different ranks
#if 1

  // Trigger sending data, flag and synchronize after
  devConn.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));

  devConn.wait();

#else
  for (int i = 1; i < world_size; i++) {
    __syncthreads();
    if (remoteRank != ((rank + i) % world_size))
      continue;

    // Trigger sending data, flag and synchronize after
    size_t ibPortion = nelemsPerGPU / 12; // nelemsPerGPU/12;
    if (isIB)
      devConn.fifo.setTrigger(trig, mscclppFlag | mscclppData | mscclppSync,
                              rank * nelemsPerGPU * sizeof(int) + (nelemsPerGPU - ibPortion) * sizeof(int),
                              rank * nelemsPerGPU * sizeof(int) + (nelemsPerGPU - ibPortion) * sizeof(int),
                              ibPortion * sizeof(int));
    else
      devConn.fifo.setTrigger(trig, mscclppFlag | mscclppData | mscclppSync, rank * nelemsPerGPU * sizeof(int),
                              rank * nelemsPerGPU * sizeof(int), (nelemsPerGPU - ibPortion) * sizeof(int));
    // Wait on the request to make sure it is safe to reuse buffer and flag
    auto req = devConn.fifo.putWithSignal(dataOffset, dataSize);
    devConn.fifo.sync(req);
  }
  // Wait for receiving data from remote rank
  while (*proxyFlag == baseFlag)
    ;
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

void print_usage(const char* prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

int main(int argc, const char* argv[])
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char* ip_port = argv[1];
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
  const char* ip_port = argv[1];
  int rank = atoi(argv[2]);
  int world_size = atoi(argv[3]);
#endif
  int localRank = rankToLocalRank(rank);
  int thisNode = rankToNode(rank);
  int cudaNum = localRank;
  int ibNum = cudaNumToIbNum(cudaNum);

  CUDACHECK(cudaSetDevice(cudaNum));
  std::string ibDevStr = "mlx5_ib" + std::to_string(localRank);

  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));

  int* data_d;
  uint64_t* flag_d;
  size_t data_size = 1536 * 1024 * 1024;
  size_t nelemsPerGPU = data_size / sizeof(int) / world_size;
  CUDACHECK(cudaMalloc(&data_d, data_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(uint64_t)));
  CUDACHECK(cudaMemset(data_d, 0, data_size));
  CUDACHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

  int* data_h = new int[nelemsPerGPU * world_size];
  for (int i = 0; i < nelemsPerGPU * world_size; i++) {
    size_t val = i + 1;
    if (i / nelemsPerGPU == rank) {
      data_h[i] = val;
    } else {
      data_h[i] = 0;
    }
  }
  CUDACHECK(cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice));

  mscclppDevConn_t devConns[16];
  for (int r = 0; r < world_size; ++r) {
    if (r == rank)
      continue;
    mscclppTransport_t transportType;
    const char* ibDev = NULL;
    transportType = mscclppTransportP2P;
    // Connect with all other ranks
    MSCCLPPCHECK(mscclppConnect(comm, &devConns[r], r, 0, data_d, data_size, flag_d, transportType, ibDev));
  }
  for (int r = 0; r < world_size; ++r) {
    if (r == rank)
      continue;
    mscclppTransport_t transportType;
    const char* ibDev = ibDevStr.c_str();
    transportType = mscclppTransportIB;
    // Connect with all other ranks
    MSCCLPPCHECK(
      mscclppConnect(comm, &devConns[r + world_size], r, 0, data_d, data_size, flag_d, transportType, ibDev));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));

  MSCCLPPCHECK(mscclppProxyLaunch(comm));

  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * 2 * world_size));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDACHECK(cudaDeviceSynchronize());
  kernel<<<1, 32 * 2 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU);
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(data_h, data_d, data_size, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (data_h[i] != val) {
      printf("oh uh things went wrong! data_h[%d] (%d) != val (%d)\n", i, data_h[i], val);
      break;
    }
  }
  int tmp[16];
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  //   // Perf test
  //   cudaEvent_t ev_start;
  //   cudaEvent_t ev_end;
  //   CUDACHECK(cudaEventCreate(&ev_start));
  //   CUDACHECK(cudaEventCreate(&ev_end));

  // warm up
  // int warmupiter = 1000;
  // for (int i = 0; i < warmupiter; ++i) {
  //   kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU);
  // }
  // CUDACHECK(cudaDeviceSynchronize());
  // MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  // cudaGraph Capture
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  int cudagraphiter = 10;
  for (int i = 0; i < cudagraphiter; ++i) {
    kernel<<<1, 32 * 2 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU);
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
  double time_in_us = ms * 1000. / (float)cudagraphlaunch / (float)cudagraphiter;
  printf("rank: %d, time: %f us/iter algBW %f\n", rank, time_in_us,
         (double)(data_size) / 1024. / 1024. / 1024. / (time_in_us / 1e6));

  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
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
