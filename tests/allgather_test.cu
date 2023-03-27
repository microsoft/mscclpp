#include "mscclpp.h"

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <unordered_map>

static int nranksPerNode = 8;

// Propagate errors up

#define MSCCLPPCHECK(call)                                                                                             \
  do {                                                                                                                 \
    mscclppResult_t res = call;                                                                                        \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      /* Print the back trace*/                                                                                        \
      printf("Failure at %s:%d -> %s\n", __FILE__, __LINE__, mscclppGetErrorString(res));                              \
      return res;                                                                                                      \
    }                                                                                                                  \
  } while (0)

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

int rankToLocalRank(int rank)
{
  return rank % nranksPerNode;
}

int rankToNode(int rank)
{
  return rank / nranksPerNode;
}

void print_usage(const char* prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

void initializeAndAllocateAllGatherData(int rank, int world_size, size_t dataSize, int nelemsPerGPU, int** data_h,
                                        int** data_d)
{
  CUDACHECK(cudaMalloc(data_d, dataSize));
  CUDACHECK(cudaMemset(*data_d, 0, dataSize));

  *data_h = new int[nelemsPerGPU * world_size];
  for (int i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (i / nelemsPerGPU == rank) {
      (*data_h)[i] = val;
    } else {
      (*data_h)[i] = 0;
    }
  }
  CUDACHECK(cudaMemcpy(*data_d, *data_h, dataSize, cudaMemcpyHostToDevice));
}

mscclppResult_t setupMscclppConnections(int rank, int world_size, mscclppComm_t comm, int* data_d, size_t dataSize)
{
  int thisNode = rankToNode(rank);
  int cudaNum = rankToLocalRank(rank);
  std::string ibDevStr = "mlx5_ib" + std::to_string(cudaNum);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank)
      continue;
    mscclppTransport_t transportType;
    const char* ibDev = ibDevStr.c_str();
    if (rankToNode(r) == thisNode) {
      ibDev = NULL;
      transportType = mscclppTransportP2P;
    } else {
      transportType = mscclppTransportIB;
    }
    // Connect with all other ranks
    MSCCLPPCHECK(mscclppConnect(comm, r, 0, data_d, dataSize, transportType, ibDev));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));

  mscclppDevConn_t* devConns;
  int nCons;
  MSCCLPPCHECK(mscclppGetAllDeviceConnections(comm, &devConns, &nCons));

  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * nCons));

  return mscclppSuccess;
}

void printUsage(const char* prog, bool isMpi)
{
  if (isMpi) {
    std::string st = "you are using MPI for this test\n";
    st += "two possilbe usages are:\n";
    st += "> " + std::string(prog) + "\n";
    st += "or\n";
    st += "> " + std::string(prog) + " -ip_port [ip:port]\n";
    printf("%s", st.c_str());
  } else {
    std::string st = "you are NOT using MPI for this test\n";
    st += "the only possible usage:\n";
    st += "> " + std::string(prog) + " -ip_port [ip:port] -rank [rank] -nranks [nranks]\n";
    printf("%s", st.c_str());
  }
}

std::unordered_map<std::string, std::string> parseArgs(int argc, const char* argv[], bool isMpi)
{
  std::unordered_map<std::string, std::string> options;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-rankspernode") {
      if (isMpi) {
        fprintf(stderr, "Error: -rankspernode should not be specified with MPI.\n");
        exit(-1);
      }
      if (i + 1 < argc) {
        options["rankspernode"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -rankspernode option requires an argument.\n");
        ;
        exit(-1);
      }
    } else if (arg == "-kernel") {
      if (i + 1 < argc) {
        options["kernel"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -kernel option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-ip_port") {
      if (i + 1 < argc) {
        options["ip_port"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -ip_port option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-rank") {
      if (isMpi) {
        fprintf(stderr, "Error: -rank should not be specified with MPI.\n");
        exit(-1);
      }
      if (i + 1 < argc) {
        options["rank"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -ip_port option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-nranks") {
      if (isMpi) {
        fprintf(stderr, "Error: -nranks should not be specified with MPI.\n");
        exit(-1);
      }
      if (i + 1 < argc) {
        options["nranks"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -ip_port option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-datasize") {
      if (i + 1 < argc) {
        options["datasize"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -datasize option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-help" || arg == "-h") {
      printUsage(argv[0], isMpi);
      exit(0);
    } else {
      fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
      exit(-1);
    }
  }
  return options;
}

int main(int argc, const char* argv[])
{
  bool isMpi = false;
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  isMpi = true;
#endif

  auto parsedArgs = parseArgs(argc, argv, isMpi);

  int rank;
  int world_size;
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // get the local number of nodes with MPI
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmrank;
  MPI_Comm_size(shmcomm, &shmrank);
  nranksPerNode = shmrank;
  MPI_Comm_free(&shmcomm);
#else
  if (parsedArgs.find("rank") == parsedArgs.end() || parsedArgs.find("nranks") == parsedArgs.end()) {
    printUsage(argv[0], isMpi);
    exit(-1);
  }
  rank = std::stoi(parsedArgs["rank"]);
  world_size = std::stoi(parsedArgs["nranks"]);
  if (parsedArgs.find("rankspernode") == parsedArgs.end()) {
    printUsage(argv[0], isMpi);
    exit(-1);
  }
  nranksPerNode = std::stoi(parsedArgs["rankspernode"]);
#endif
  int kernelNum = 0;
  if (parsedArgs.find("kernel") != parsedArgs.end()) {
    kernelNum = std::stoi(parsedArgs["kernel"]);
  }
  char* ip_port = NULL;
  if (parsedArgs.find("ip_port") == parsedArgs.end()) {
    printUsage(argv[0], isMpi);
    exit(-1);
  }
  ip_port = (char*)parsedArgs["ip_port"].c_str();

  int thisNode = rankToNode(rank);
  int cudaNum = rankToLocalRank(rank);
  CUDACHECK(cudaSetDevice(cudaNum));

  if (rank == 0)
    printf("Initializing MSCCL++\n");
  mscclppComm_t comm;
  MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, ip_port, rank));

  int* data_d;
  int* data_h;
  size_t dataSize = 1024 * 1024 * 1024;
  if (parsedArgs.find("datasize") != parsedArgs.end()) {
    dataSize = std::stoi(parsedArgs["datasize"]);
  }
  int nelemsPerGPU = dataSize / sizeof(int) / world_size;

  if (rank == 0)
    printf("Initializing data for allgather test\n");
  initializeAndAllocateAllGatherData(rank, world_size, dataSize, nelemsPerGPU, &data_h, &data_d);

  if (rank == 0)
    printf("Setting up the connection in MSCCL++\n");
  MSCCLPPCHECK(setupMscclppConnections(rank, world_size, comm, data_d, dataSize));

  if (rank == 0)
    printf("Launching MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyLaunch(comm));

  if (rank == 0)
    printf("Testing the correctness of AllGather implementation\n");
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaDeviceSynchronize());
  kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU, kernelNum);
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(data_h, data_d, dataSize, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

  for (int i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (data_h[i] != val) {
      printf("oh uh! data_h[%d] (%d) != val (%d)\n", i, data_h[i], val);
      break;
    }
  }
  int tmp[16];
  // A simple barrier
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
  if (rank == 0)
    printf("Successfully checked the correctness\n");

  // Perf test
  int iterwithoutcudagraph = 10;
  if (rank == 0)
    printf("Running %d iterations of the kernel without CUDA graph\n", iterwithoutcudagraph);
  for (int i = 0; i < iterwithoutcudagraph; ++i) {
    kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU, kernelNum);
  }
  CUDACHECK(cudaDeviceSynchronize());
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  // cudaGraph Capture
  int cudagraphiter = 10;
  if (rank == 0)
    printf("Capturing %d iterations of the kernel in a CUDA graph\n", cudagraphiter);
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < cudagraphiter; ++i) {
    kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nelemsPerGPU, kernelNum);
  }
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

  int cudagraphwarmup = 10;
  if (rank == 0)
    printf("Warming up %d iterations of the CUDA graph with %d iterations of the kernel\n", cudagraphwarmup,
           cudagraphiter);
  for (int i = 0; i < cudagraphwarmup; ++i) {
    cudaGraphLaunch(instance, stream);
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  // measure runtime
  int cudagraphlaunch = 10;
  if (rank == 0)
    printf("Running %d iterations of the CUDA graph with %d iterations of the kernel\n", cudagraphlaunch,
           cudagraphiter);
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
  double t0 = getTime();
  for (int i = 0; i < cudagraphlaunch; ++i) {
    cudaGraphLaunch(instance, stream);
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  double t1 = getTime();
  float ms = (t1 - t0) * 1000.0;
  double time_in_us = ms * 1000. / (float)cudagraphlaunch / (float)cudagraphiter;
  printf("Rank %d report: size %lu time: %f us/iter algBW %f GBps\n", rank, dataSize, time_in_us,
         (double)(dataSize) / 1e9 / (time_in_us / 1e6));
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  if (rank == 0)
    printf("Stopping MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyStop(comm));

  if (rank == 0)
    printf("Destroying MSCCL++ communicator\n");
  MSCCLPPCHECK(mscclppCommDestroy(comm));
  printf("Rank %d succeeded!\n", rank);

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return 0;
}
