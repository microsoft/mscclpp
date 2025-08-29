// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>
#include <sys/wait.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <unordered_map>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

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

class AllgatherAlgo : public std::enable_shared_from_this<AllgatherAlgo> {
 public:
  AllgatherAlgo() = default;
  ~AllgatherAlgo() {
    if (proxyService_) {
      proxyService_->stopProxy();
    }
  }

  void registerAlgorithm() {
    auto self = shared_from_this();
    mscclpp::Algorithm allgatherAlgo(
        "allgather", [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
        [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count,
               int dtype, cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
          return self->allgatherKernelFunc(ctx, input, output, count, static_cast<ncclDataType_t>(dtype), stream,
                                           extras);
        },
        [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype) {
          return self->initAllgatherContext(comm, input, output, count, static_cast<ncclDataType_t>(dtype));
        },
        [self](const void* input, void* output, size_t count, int dtype) {
          return self->generateAllgatherContextKey(input, output, count, static_cast<ncclDataType_t>(dtype));
        });
    mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("allgather", "allgather", allgatherAlgo);
  }

 private:
  std::vector<std::shared_ptr<mscclpp::Connection>> conns_;
  std::shared_ptr<mscclpp::ProxyService> proxyService_;
  int worldSize_;

  void initialize(std::shared_ptr<mscclpp::Communicator> comm) {
    std::vector<std::shared_future<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
    worldSize_ = comm->bootstrap()->getNranks();
    for (int i = 0; i < worldSize_; i++) {
      if (i == comm->bootstrap()->getRank()) continue;
      connectionFutures.push_back(comm->connect(mscclpp::Transport::CudaIpc, i));
    }
    std::vector<std::shared_ptr<mscclpp::Connection>> connections;
    std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                   [](const auto& future) { return future.get(); });
    this->conns_ = std::move(connections);
    proxyService_ = std::make_shared<mscclpp::ProxyService>();
    proxyService_->startProxy();
  }

  ncclResult_t allgatherKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output,
                                   size_t count, [[maybe_unused]] ncclDataType_t dtype, cudaStream_t stream,
                                   std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
    int rank = ctx->rank;
    int worldSize = ctx->workSize;

    int nThreadsPerBlock = (worldSize - 1) * WARP_SIZE;
    allgather<<<1, nThreadsPerBlock, 0, stream>>>(ctx->portChannelDeviceHandles.get(), rank,
                                                  count * ncclTypeSize(dtype));
    if (cudaGetLastError() == cudaSuccess) {
      return ncclSuccess;
    }
    return ncclInternalError;
  }

  std::shared_ptr<mscclpp::AlgorithmCtx> initAllgatherContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                              const void* input, void* output, size_t count,
                                                              ncclDataType_t dtype) {
    auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
    ctx->rank = comm->bootstrap()->getRank();
    ctx->workSize = comm->bootstrap()->getNranks();
    ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

    // register memories
    mscclpp::RegisteredMemory inputBufRegMem =
        comm->registerMemory((void*)input, count * ncclTypeSize(dtype), mscclpp::Transport::CudaIpc);
    mscclpp::RegisteredMemory outputBufRegMem =
        comm->registerMemory(output, count * ncclTypeSize(dtype) * ctx->workSize, mscclpp::Transport::CudaIpc);
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

  mscclpp::AlgorithmCtxKey generateAllgatherContextKey(const void* input, void* output, size_t count,
                                                       ncclDataType_t dtype) {
    return {(void*)input, output, count * ncclTypeSize(dtype), count * ncclTypeSize(dtype) * worldSize_, 0};
  }
};

void worker(int rank, int worldSize, ncclUniqueId id) {
  constexpr int size = 1024 * 1024;
  MSCCLPP_CUDATHROW(cudaSetDevice(rank));
  auto allgatherAlgo = std::make_shared<AllgatherAlgo>();
  allgatherAlgo->registerAlgorithm();

  auto algoFactory = mscclpp::AlgorithmFactory::getInstance();
  algoFactory->setAlgorithmSelector(
      [](const std::unordered_map<std::string, std::unordered_map<std::string, mscclpp::Algorithm>>&
             algoMapByCollective,
         std::string collective, size_t messageSize, int nRanksPerNode, int worldSize) {
        if (collective != "allgather") {
          return mscclpp::Algorithm();
        }
        return algoMapByCollective.at(collective).at("allgather");
      });

  float *sendbuff, *recvbuff;
  cudaStream_t s;
  MSCCLPP_CUDATHROW(cudaMalloc(&sendbuff, size * sizeof(float)));
  MSCCLPP_CUDATHROW(cudaMalloc(&recvbuff, size * sizeof(float) * worldSize));
  MSCCLPP_CUDATHROW(cudaStreamCreate(&s));

  ncclComm_t comm;
  ncclCommInitRank(&comm, worldSize, id, rank);
  ncclAllGather(sendbuff, recvbuff, size, ncclFloat, comm, s);

  MSCCLPP_CUDATHROW(cudaStreamSynchronize(s));

  MSCCLPP_CUDATHROW(cudaFree(sendbuff));
  MSCCLPP_CUDATHROW(cudaFree(recvbuff));

  ncclCommDestroy(comm);
}

int main() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  int pid0 = spawn_process([&]() { worker(0, 4, id); });
  int pid1 = spawn_process([&]() { worker(1, 4, id); });
  int pid2 = spawn_process([&]() { worker(2, 4, id); });
  int pid3 = spawn_process([&]() { worker(3, 4, id); });

  wait_process(pid0);
  wait_process(pid1);
  wait_process(pid2);
  wait_process(pid3);
  return 0;
}