// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/ext/nccl/nccl.h>
#include <sys/wait.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <unordered_map>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <typename... Args>
void log(Args&&... args) {
  std::stringstream ss;
  (ss << ... << args);
  ss << std::endl;
  std::cout << ss.str();
}

int spawn_process(std::function<void()> func) {
  pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    // Child process
    func();
    exit(0);
  }
  return pid;
}

int wait_process(int pid) {
  int status;
  if (waitpid(pid, &status, 0) < 0) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
}

__global__ void __launch_bounds__(1024)
    allgather(mscclpp::DeviceHandle<mscclpp::PortChannel>* portChannels, int rank, size_t nbytesPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  mscclpp::DeviceHandle<mscclpp::PortChannel> portChan = portChannels[warpId];

  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.putWithSignal(rank * nbytesPerGPU, 0, nbytesPerGPU);
  }
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.flush();
  }

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.wait();
  }
}

class AllgatherAlgoBuilder : public mscclpp::AlgorithmBuilder {
 public:
  AllgatherAlgoBuilder() = default;
  ~AllgatherAlgoBuilder() {
    if (proxyService_) {
      proxyService_->stopProxy();
    }
  }

  std::shared_ptr<mscclpp::Algorithm> build() override {
    auto self = std::make_shared<AllgatherAlgoBuilder>();
    std::shared_ptr<mscclpp::Algorithm> allgatherAlgo = std::make_shared<mscclpp::NativeAlgorithm>(
        "allgather", "allgather", [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
        [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
               size_t outputSize, mscclpp::DataType dtype, [[maybe_unused]] mscclpp::ReduceOp op, cudaStream_t stream,
               int nBlocks, int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras) {
          return self->allgatherKernelFunc(ctx, input, output, inputSize, stream);
        },
        [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
               size_t outputSize,
               mscclpp::DataType dtype) { return self->initAllgatherContext(comm, input, output, inputSize, dtype); },
        [self](const void* input, void* output, size_t inputSize, size_t outputSize, mscclpp::DataType dtype) {
          return self->generateAllgatherContextKey(input, output, inputSize, outputSize,
                                                   static_cast<ncclDataType_t>(dtype));
        });
    return allgatherAlgo;
  }

 private:
  std::vector<mscclpp::Connection> conns_;
  std::shared_ptr<mscclpp::ProxyService> proxyService_;
  int worldSize_;

  void initialize(std::shared_ptr<mscclpp::Communicator> comm) {
    std::vector<std::shared_future<mscclpp::Connection>> connectionFutures;
    worldSize_ = comm->bootstrap()->getNranks();
    for (int i = 0; i < worldSize_; i++) {
      if (i == comm->bootstrap()->getRank()) continue;
      connectionFutures.push_back(comm->connect(mscclpp::Transport::CudaIpc, i));
    }
    std::vector<mscclpp::Connection> connections;
    std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                   [](const auto& future) { return future.get(); });
    this->conns_ = std::move(connections);
    proxyService_ = std::make_shared<mscclpp::ProxyService>();
    proxyService_->startProxy();
  }

  mscclpp::CommResult allgatherKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                          void* output, size_t inputSize, cudaStream_t stream) {
    int rank = ctx->rank;
    int worldSize = ctx->workSize;

    int nThreadsPerBlock = (worldSize - 1) * WARP_SIZE;
    allgather<<<1, nThreadsPerBlock, 0, stream>>>(ctx->portChannelDeviceHandles.get(), rank, inputSize);
    if (cudaGetLastError() == cudaSuccess) {
      return mscclpp::CommResult::CommSuccess;
    }
    return mscclpp::CommResult::CommInternalError;
  }

  std::shared_ptr<mscclpp::AlgorithmCtx> initAllgatherContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                              const void* input, void* output, size_t inputSize,
                                                              mscclpp::DataType dtype) {
    auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
    ctx->rank = comm->bootstrap()->getRank();
    ctx->workSize = comm->bootstrap()->getNranks();
    ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

    // register memories
    mscclpp::RegisteredMemory inputBufRegMem =
        comm->registerMemory((void*)input, inputSize, mscclpp::Transport::CudaIpc);
    mscclpp::RegisteredMemory outputBufRegMem =
        comm->registerMemory(output, inputSize * ctx->workSize, mscclpp::Transport::CudaIpc);
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRegMemories;
    for (int i = 0; i < ctx->workSize; i++) {
      if (i == ctx->rank) continue;
      comm->sendMemory(outputBufRegMem, i, 0);
      remoteRegMemories.push_back(comm->recvMemory(i, 0));
    }

    // setup channels
    std::vector<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannels;
    mscclpp::MemoryId inputMemoryId = this->proxyService_->addMemory(inputBufRegMem);
    for (int i = 0; i < this->conns_.size(); i++) {
      auto remoteMemory = remoteRegMemories[i].get();
      mscclpp::MemoryId remoteMemoryId = this->proxyService_->addMemory(remoteMemory);
      portChannels.push_back(mscclpp::deviceHandle(this->proxyService_->portChannel(
          this->proxyService_->buildAndAddSemaphore(*comm, this->conns_[i]), remoteMemoryId, inputMemoryId)));
    }
    ctx->portChannelDeviceHandles =
        mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::PortChannel>>(portChannels.size());
    mscclpp::gpuMemcpy(ctx->portChannelDeviceHandles.get(), portChannels.data(), portChannels.size(),
                       cudaMemcpyHostToDevice);

    // keep registered memory references
    std::transform(remoteRegMemories.begin(), remoteRegMemories.end(), std::back_inserter(ctx->registeredMemories),
                   [](const auto& fut) { return fut.get(); });
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

    return ctx;
  }

  mscclpp::AlgorithmCtxKey generateAllgatherContextKey(const void* input, void* output, size_t inputSize,
                                                       size_t outputSize, ncclDataType_t dtype) {
    return {(void*)input, output, inputSize, outputSize, 0};
  }
};

void worker(int rank, int worldSize, ncclUniqueId id) {
  constexpr int size = 1024 * 1024 * 64;
  const int iter = 100;
  MSCCLPP_CUDATHROW(cudaSetDevice(rank));

  // register algorithm
  auto allgatherAlgoBuilder = std::make_shared<AllgatherAlgoBuilder>();
  auto algoCollectionBuilder = mscclpp::collective::AlgorithmCollectionBuilder::getInstance();
  algoCollectionBuilder->addAlgorithmBuilder(allgatherAlgoBuilder);
  algoCollectionBuilder->setAlgorithmSelector(
      [](const std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<mscclpp::Algorithm>>>&
             algoMapByCollective,
         const mscclpp::CollectiveRequest& request) -> std::shared_ptr<mscclpp::Algorithm> {
        if (request.collective != "allgather") {
          return nullptr;
        }
        return algoMapByCollective.at(request.collective).at("allgather");
      });

  float *sendbuff, *recvbuff;
  cudaStream_t stream;
  MSCCLPP_CUDATHROW(cudaMalloc(&sendbuff, size * sizeof(float)));
  MSCCLPP_CUDATHROW(cudaMalloc(&recvbuff, size * sizeof(float) * worldSize));
  MSCCLPP_CUDATHROW(cudaMemcpy(recvbuff + rank * size, sendbuff, size * sizeof(float), cudaMemcpyDeviceToDevice));
  MSCCLPP_CUDATHROW(cudaStreamCreate(&stream));

  ncclComm_t comm;
  cudaGraphExec_t graphExec;
  cudaGraph_t graph;
  MSCCLPP_CUDATHROW(cudaGraphCreate(&graph, 0));

  ncclCommInitRank(&comm, worldSize, id, rank);
  MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < iter; ++i) {
    ncclAllGather(sendbuff, recvbuff, size, ncclFloat, comm, stream);
  }
  MSCCLPP_CUDATHROW(cudaStreamEndCapture(stream, &graph));
  MSCCLPP_CUDATHROW(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaEvent_t start, end;
  if (rank == 0) {
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  }

  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  if (rank == 0) {
    MSCCLPP_CUDATHROW(cudaEventRecord(start, stream));
  }
  MSCCLPP_CUDATHROW(cudaGraphLaunch(graphExec, stream));
  if (rank == 0) {
    MSCCLPP_CUDATHROW(cudaEventRecord(end, stream));
    MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
    float elapsedTime;
    float elapsedTimePerIter;
    float gbps;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedTime, start, end));
    elapsedTimePerIter = elapsedTime / iter;
    gbps = float(size) * (worldSize - 1) * ncclTypeSize(ncclFloat) / elapsedTimePerIter * 1e-6f;
    log("GPU ", rank, ": bytes ", size * ncclTypeSize(ncclFloat), ", elapsed ", elapsedTimePerIter, " ms/iter, BW ",
        gbps, " GB/s");
  }

  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  MSCCLPP_CUDATHROW(cudaFree(sendbuff));
  MSCCLPP_CUDATHROW(cudaFree(recvbuff));

  ncclCommDestroy(comm);
  algoCollectionBuilder->reset();
}

int main() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  int pid0 = spawn_process([&]() { worker(0, 4, id); });
  int pid1 = spawn_process([&]() { worker(1, 4, id); });
  int pid2 = spawn_process([&]() { worker(2, 4, id); });
  int pid3 = spawn_process([&]() { worker(3, 4, id); });

  if (pid0 < 0 || pid1 < 0 || pid2 < 0 || pid3 < 0) {
    log("Fork failed!");
    return -1;
  }

  int status0 = wait_process(pid0);
  int status1 = wait_process(pid1);
  int status2 = wait_process(pid2);
  int status3 = wait_process(pid3);
  if (status0 != 0 || status1 != 0 || status2 != 0 || status3 != 0) {
    log("Worker failed!");
    return -1;
  }

  log("Succeed!");
  return 0;
}