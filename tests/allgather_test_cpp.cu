#include "mscclpp.h"
#include "mscclpp.hpp"

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

__constant__ mscclpp::DeviceConnection constDevConns[16];

__device__ void allgather0(mscclppDevConn_t devConn, int rank, int world_size, int remoteRank, size_t nelemsPerGPU)
{
  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if ((threadIdx.x % 32) == 0)
    devConn.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if ((threadIdx.x % 32) == 0)
    devConn.flush();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if ((threadIdx.x % 32) == 0)
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

void initializeAndAllocateAllGatherData(int rank, int world_size, size_t dataSize, size_t nelemsPerGPU, int** data_h,
                                        int** data_d)
{
  CUDACHECK(cudaMalloc(data_d, dataSize));
  CUDACHECK(cudaMemset(*data_d, 0, dataSize));

  *data_h = new int[nelemsPerGPU * world_size];
  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (i / nelemsPerGPU == (size_t)rank) {
      (*data_h)[i] = val;
    } else {
      (*data_h)[i] = 0;
    }
  }
  CUDACHECK(cudaMemcpy(*data_d, *data_h, dataSize, cudaMemcpyHostToDevice));
}

void setupMscclppConnections(int rank, int world_size, mscclpp::Communicator& comm, int* data_d, size_t dataSize)
{
  int thisNode = rankToNode(rank);
  int cudaNum = rankToLocalRank(rank);
  std::string ibDevStr = "mlx5_ib" + std::to_string(cudaNum);
  std::vector<mscclpp::DeviceConnection> devConns(world_size);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank)
      continue;
    mscclpp::TransportType transportType;
    const char* ibDev = ibDevStr.c_str();
    if (rankToNode(r) == thisNode) {
      ibDev = NULL;
      transportType = mscclpp::TransportType::P2P;
    } else {
      transportType = mscclpp::TransportType::IB;
    }
    // Connect with all other ranks
    auto hostConn = comm.connect(r, 0, transportType, ibDev);
    hostConn->registerBuffer(data_d, dataSize);
    devConns.push_back(hostConn->toDevice(false));
  }

  comm.connectionSetup();
  assert(devConns.size() < sizeof(constDevConns) / sizeof(mscclpp::DeviceConnection));
  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns.data(), sizeof(mscclpp::DeviceConnection) * devConns.size() ));
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

  int* data_d;
  int* data_h;
  size_t dataSize = 1024 * 1024 * 1024;
  if (parsedArgs.find("datasize") != parsedArgs.end()) {
    dataSize = std::stoul(parsedArgs["datasize"]);
  }
  size_t nelemsPerGPU = dataSize / sizeof(int) / world_size;

  try{
    mscclpp::Communicator comm;

    if (rank == 0)
        printf("Initializing MSCCL++\n");

    comm.initRank(world_size, ip_port, rank);

    if (rank == 0)
        printf("Initializing data for allgather test\n");
    initializeAndAllocateAllGatherData(rank, world_size, dataSize, nelemsPerGPU, &data_h, &data_d);

    if (rank == 0)
        printf("Setting up the connection in MSCCL++\n");
    setupMscclppConnections(rank, world_size, comm, data_d, dataSize);

  } catch (std::exception& e) {
    // todo: throw exceptions in the implementation and process them here
  }

  if (rank == 0)
    printf("Launching MSCCL++ proxy threads\n");
  MSCCLPPCHECK(mscclppProxyLaunch(comm));

  if (rank == 0)
    printf("Testing the correctness of AllGather implementation\n");
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaDeviceSynchronize());
  kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nranksPerNode, nelemsPerGPU, kernelNum);
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(data_h, data_d, dataSize, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (data_h[i] != val) {
      printf("oh uh! data_h[%ld] (%d) != val (%d)\n", i, data_h[i], val);
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
  CUDACHECK(cudaStreamSynchronize(stream));
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));
  for (int i = 0; i < iterwithoutcudagraph; ++i) {
    kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nranksPerNode, nelemsPerGPU, kernelNum);
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  MSCCLPPCHECK(mscclppBootstrapAllGather(comm, tmp, sizeof(int)));

  // cudaGraph Capture
  int cudagraphiter = 10;
  if (rank == 0)
    printf("Capturing %d iterations of the kernel in a CUDA graph\n", cudagraphiter);
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < cudagraphiter; ++i) {
    kernel<<<1, 32 * (world_size - 1), 0, stream>>>(rank, world_size, nranksPerNode, nelemsPerGPU, kernelNum);
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
  double t0, t1, ms, time_in_us;
  t0 = getTime();
  for (int i = 0; i < cudagraphlaunch; ++i) {
    cudaGraphLaunch(instance, stream);
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  t1 = getTime();
  ms = (t1 - t0) * 1000.0;
  time_in_us = ms * 1000. / (float)cudagraphlaunch / (float)cudagraphiter;
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