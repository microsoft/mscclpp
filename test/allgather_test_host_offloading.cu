#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/epoch.hpp>

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <unordered_map>

int nranksPerNode;
int rank;
int world_size;

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


__global__ void kernel(int r, int nranks, mscclpp::DeviceProxyFifo fifo, mscclpp::DeviceEpoch::DeviceHandle* handles)
{
  int tid = threadIdx.x;
  if (tid != r)
    handles[tid].epochIncrement();
  if (tid == 0){
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 1;
    fifo.push(trigger);
  }
  if (tid != r)
    handles[tid].wait();
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

class MyProxyService {
private:
  int deviceNumaNode;
public:
  MyProxyService() : remoteMemories(world_size), connections(world_size),
  proxy([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
    // int cudaDevice;
    // CUDACHECK(cudaGetDevice(&cudaDevice));
    // getDeviceNumaNode(cudaDevice, &deviceNumaNode);
  }

  void bindThread() {
    // if (deviceNumaNode >= 0) {
    //   numaBind(deviceNumaNode);
    //   INFO(MSCCLPP_INIT, "NUMA node of DeviceChannelService proxy thread is set to %d", deviceNumaNode);
    // }
  }

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
    if (triggerRaw.fst == 1) {
      int dataSizePerRank = dataSize / world_size;
      for (int r = 0; r < world_size; ++r) {
        if (r == rank) {
          continue;
        }
        connections[r]->write(remoteMemories[r], rank*dataSizePerRank, localMemory, rank*dataSizePerRank, dataSizePerRank);
        deviceEpochs[r]->signal();
        connections[r]->flush();
      }
    }
    return mscclpp::ProxyHandlerResult::FlushFifoTailAndContinue;
  }
  mscclpp::Proxy proxy;
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  mscclpp::RegisteredMemory localMemory;
  std::vector<std::shared_ptr<mscclpp::HostEpoch>> hostEpochs;
  std::vector<std::shared_ptr<mscclpp::DeviceEpoch>> deviceEpochs;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  int dataSize; 
};

void setupProxyService(mscclpp::Communicator& comm, MyProxyService& proxyService, int* data_d, int dataSize)
{
  proxyService.dataSize = dataSize;
  int thisNode = rankToNode(rank);
  int cudaNum = rankToLocalRank(rank);
  std::string ibDevStr = "mlx5_ib" + std::to_string(cudaNum);
  mscclpp::Transport ibTransport = mscclpp::getIBTransportByDeviceName(ibDevStr);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemories(world_size);

  proxyService.localMemory = comm.registerMemory(data_d, dataSize, mscclpp::Transport::CudaIpc | ibTransport);
  for (int r = 0; r < world_size; ++r) {
    if (r == rank){
      proxyService.hostEpochs.emplace_back(nullptr);
      proxyService.deviceEpochs.emplace_back(nullptr);
      continue;
    }
    mscclpp::Transport transport;
    if (rankToNode(r) == thisNode) {
      transport = mscclpp::Transport::CudaIpc;
    } else {
      transport = ibTransport;
    }
    // Connect with all other ranks
    proxyService.connections[r] = comm.connectOnSetup(r, 0, transport);
    if (rankToNode(r) == thisNode) {
      proxyService.hostEpochs.emplace_back(nullptr);
    } else {
      proxyService.hostEpochs.emplace_back(std::make_shared<mscclpp::HostEpoch>(comm, proxyService.connections[r]));
    }
    proxyService.deviceEpochs.emplace_back(std::make_shared<mscclpp::DeviceEpoch>(comm, proxyService.connections[r]));
    comm.sendMemoryOnSetup(proxyService.localMemory, r, 0);

    remoteMemories[r] = comm.recvMemoryOnSetup(r, 0);
  }

  comm.setup();
  for (int r = 0; r < world_size; ++r) {
    if (r == rank){
      continue;
    }
    proxyService.remoteMemories[r] = remoteMemories[r].get();
  }
}

std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[])
{
  std::unordered_map<std::string, std::string> options;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-datasize") {
      if (i + 1 < argc) {
        options["datasize"] = argv[++i];
      } else {
        fprintf(stderr, "Error: -datasize option requires an argument.\n");
        exit(-1);
      }
    } else if (arg == "-help" || arg == "-h") {
      exit(0);
    } else {
      fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
      exit(-1);
    }
  }
  return options;
}


int main(int argc, char* argv[])
{
  // sleep(10);
  MPI_Init(&argc, &argv);
  auto parsedArgs = parseArgs(argc, argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // get the local number of nodes with MPI
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  int shmrank;
  MPI_Comm_size(shmcomm, &shmrank);
  nranksPerNode = shmrank;
  MPI_Comm_free(&shmcomm);


  int cudaNum = rankToLocalRank(rank);
  CUDACHECK(cudaSetDevice(cudaNum));

  if (rank == 0)
    printf("Initializing MSCCL++\n");
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(rank, world_size);
  mscclpp::UniqueId uniqueId;
  if (rank == 0)
    uniqueId = bootstrap->createUniqueId();
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(uniqueId);
  mscclpp::Communicator comm(bootstrap);

  int* data_d;
  int* data_h;
  size_t dataSize = 1024 * 1024 * 1024;
  if (parsedArgs.find("datasize") != parsedArgs.end()) {
    dataSize = std::stoul(parsedArgs["datasize"]);
  }
  size_t nelemsPerGPU = dataSize / sizeof(int) / world_size;

  if (rank == 0)
    printf("Initializing data for allgather test\n");
  initializeAndAllocateAllGatherData(rank, world_size, dataSize, nelemsPerGPU, &data_h, &data_d);

  if (rank == 0)
    printf("Setting up the connection in MSCCL++\n");

  MyProxyService proxyService;
  setupProxyService(comm, proxyService, data_d, dataSize);

  if (rank == 0)
    printf("Launching MSCCL++ proxy threads\n");
  proxyService.proxy.start();
  mscclpp::DeviceProxyFifo fifo = proxyService.proxy.fifo().deviceFifo();
  if (rank == 0)
    printf("Testing the correctness of AllGather implementation\n");
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  mscclpp::DeviceEpoch::DeviceHandle* deviceHandles;

  CUDACHECK(cudaMalloc(&deviceHandles, sizeof(mscclpp::DeviceEpoch::DeviceHandle) * world_size));
  for (int i = 0; i < world_size; ++i) {
    if (i == rank)
      continue;
    auto handle = proxyService.deviceEpochs[i]->deviceHandle();
    CUDACHECK(cudaMemcpy(&deviceHandles[i], &handle, sizeof(mscclpp::DeviceEpoch::DeviceHandle), cudaMemcpyHostToDevice));
  }

  kernel<<<1, world_size, 0, stream>>>(rank, world_size, fifo, deviceHandles);
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaMemcpy(data_h, data_d, dataSize, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nelemsPerGPU * world_size; i++) {
    int val = i + 1;
    if (data_h[i] != val) {
      printf("oh uh! data_h[%ld] (%d) != val (%d)\n", i, data_h[i], val);
      break;
    }
  }

  bootstrap->barrier();
  if (rank == 0)
    printf("Correctness test passed!\n");

  double t0, t1, ms, time_in_us;
  int iterwithoutcudagraph = 10;
  if (rank == 0)
    printf("Running %d iterations of the kernel without CUDA graph\n", iterwithoutcudagraph);
  CUDACHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  t0 = getTime();
  for (int i = 0; i < iterwithoutcudagraph; ++i) {
    kernel<<<1, world_size, 0, stream>>>(rank, world_size, fifo, deviceHandles);
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  bootstrap->barrier();
  t1 = getTime();
  ms = (t1 - t0) * 1000.0;
  time_in_us = ms * 1000. / (float)iterwithoutcudagraph;
  printf("No Graph %d report: size %lu time: %f us/iter algBW %f GBps\n", rank, dataSize, time_in_us,
          (double)(dataSize) / 1e9 / (time_in_us / 1e6));

  // cudaGraph Capture
  int cudagraphiter = 10;
  if (rank == 0)
    printf("Capturing %d iterations of the kernel in a CUDA graph\n", cudagraphiter);
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < cudagraphiter; ++i) {
    kernel<<<1, world_size, 0, stream>>>(rank, world_size, fifo, deviceHandles);
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
  bootstrap->barrier();
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
  bootstrap->barrier();

  if (rank == 0)
    printf("Stopping MSCCL++ proxy threads\n");
  proxyService.proxy.stop();



#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  MPI_Finalize();
#endif
  return 0;
}
