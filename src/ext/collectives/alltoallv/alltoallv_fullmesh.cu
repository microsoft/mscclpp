// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "alltoallv/alltoallv_fullmesh.hpp"
#include "alltoallv/alltoallv_kernel.hpp"

#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/gpu_utils.hpp>

#include <algorithm>

namespace mscclpp {
namespace collective {

#if defined(__HIP_PLATFORM_AMD__)
#define ALLTOALLV_WARP_SIZE 64
#else
#define ALLTOALLV_WARP_SIZE 32
#endif

// Context to hold all necessary state for alltoallv execution
struct AllToAllVContext {
  int rank;
  int worldSize;
  int nRanksPerNode;

  std::vector<RegisteredMemory> registeredMemories;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
};

AlltoallvFullmesh::~AlltoallvFullmesh() {
  if (proxyService_) {
    proxyService_->stopProxy();
  }
}

std::shared_ptr<Algorithm> AlltoallvFullmesh::build() {
  auto self = std::shared_ptr<AlltoallvFullmesh>(this, [](AlltoallvFullmesh*) {});

  std::shared_ptr<Algorithm> alltoallvAlgo = std::make_shared<NativeAlgorithm>(
      "alltoallv", "alltoallv_fullmesh",
      // Initialize function
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      // Kernel execution function
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             size_t outputSize, DataType dtype, [[maybe_unused]] ReduceOp op, cudaStream_t stream,
             int nBlocks, int nThreadsPerBlock,
             const std::unordered_map<std::string, uintptr_t>& extras) {
        return self->alltoallvKernelFunc(ctx, input, output, inputSize, outputSize, dtype, stream,
                                         nBlocks, nThreadsPerBlock, extras);
      },
      // Context initialization function
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             size_t outputSize, DataType dtype) {
        return self->initAlltoallvContext(comm, input, output, inputSize, outputSize, dtype);
      },
      // Context key generation function
      [self](const void* input, void* output, size_t inputSize, size_t outputSize, DataType dtype) {
        return self->generateAlltoallvContextKey(input, output, inputSize, outputSize, dtype);
      });

  return alltoallvAlgo;
}

void AlltoallvFullmesh::initialize(std::shared_ptr<Communicator> comm) {
  std::vector<std::shared_future<Connection>> connectionFutures;
  worldSize_ = comm->bootstrap()->getNranks();
  int rank = comm->bootstrap()->getRank();

  for (int i = 0; i < worldSize_; i++) {
    if (i == rank) continue;
    connectionFutures.push_back(comm->connect(Transport::CudaIpc, i));
  }

  std::vector<Connection> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });
  this->conns_ = std::move(connections);

  proxyService_ = std::make_shared<ProxyService>();
  proxyService_->startProxy(true);
}

CommResult AlltoallvFullmesh::alltoallvKernelFunc(
    const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
    size_t outputSize, [[maybe_unused]] DataType dtype, cudaStream_t stream,
    [[maybe_unused]] int nBlocks, [[maybe_unused]] int nThreadsPerBlock,
    const std::unordered_map<std::string, uintptr_t>& extras) {

  auto algoCtx = std::static_pointer_cast<AllToAllVContext>(ctx);
  int rank = algoCtx->rank;
  int worldSize = algoCtx->worldSize;

  // Extract send/recv counts and displacements from extras
  auto it_sendCounts = extras.find("sendCounts");
  auto it_sendDispls = extras.find("sendDispls");
  auto it_recvCounts = extras.find("recvCounts");
  auto it_recvDispls = extras.find("recvDispls");

  if (it_sendCounts == extras.end() || it_sendDispls == extras.end() ||
      it_recvCounts == extras.end() || it_recvDispls == extras.end()) {
    return CommResult::CommInternalError;
  }

  const size_t* d_sendCounts = reinterpret_cast<const size_t*>(it_sendCounts->second);
  const size_t* d_sendDispls = reinterpret_cast<const size_t*>(it_sendDispls->second);
  const size_t* d_recvCounts = reinterpret_cast<const size_t*>(it_recvCounts->second);
  const size_t* d_recvDispls = reinterpret_cast<const size_t*>(it_recvDispls->second);

  // Choose kernel based on world size
  if (worldSize <= 16) {
    // Use parallel warp-based kernel for small world sizes
    int nThreads = (worldSize - 1) * ALLTOALLV_WARP_SIZE;
    if (nThreads < 32) nThreads = 32;
    if (nThreads > 1024) nThreads = 1024;

    alltoallvKernel<<<1, nThreads, 0, stream>>>(
        algoCtx->portChannelDeviceHandles.get(),
        rank, worldSize,
        input, output,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls);
  } else {
    // Use ring-based kernel for larger world sizes
    alltoallvRingKernel<<<1, 32, 0, stream>>>(
        algoCtx->portChannelDeviceHandles.get(),
        rank, worldSize,
        input, output,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls);
  }

  if (cudaGetLastError() == cudaSuccess) {
    return CommResult::CommSuccess;
  }
  return CommResult::CommInternalError;
}

std::shared_ptr<void> AlltoallvFullmesh::initAlltoallvContext(
    std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
    size_t outputSize, [[maybe_unused]] DataType dtype) {

  auto ctx = std::make_shared<AllToAllVContext>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->worldSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // Register memories for input and output buffers
  RegisteredMemory inputBufRegMem = comm->registerMemory((void*)input, inputSize, Transport::CudaIpc);
  RegisteredMemory outputBufRegMem = comm->registerMemory(output, outputSize, Transport::CudaIpc);

  // Exchange output buffer registration with all peers
  std::vector<std::shared_future<RegisteredMemory>> remoteRegMemories;
  for (int i = 0; i < ctx->worldSize; i++) {
    if (i == ctx->rank) continue;
    comm->sendMemory(outputBufRegMem, i, 0);
    remoteRegMemories.push_back(comm->recvMemory(i, 0));
  }

  // Setup port channels for each peer
  std::vector<DeviceHandle<PortChannel>> portChannels;
  MemoryId inputMemoryId = this->proxyService_->addMemory(inputBufRegMem);

  for (size_t i = 0; i < this->conns_.size(); i++) {
    auto remoteMemory = remoteRegMemories[i].get();
    MemoryId remoteMemoryId = this->proxyService_->addMemory(remoteMemory);
    portChannels.push_back(deviceHandle(this->proxyService_->portChannel(
        this->proxyService_->buildAndAddSemaphore(*comm, this->conns_[i]), remoteMemoryId, inputMemoryId)));
  }

  // Allocate and copy port channels to device
  ctx->portChannelDeviceHandles = detail::gpuCallocShared<DeviceHandle<PortChannel>>(portChannels.size());
  gpuMemcpy(ctx->portChannelDeviceHandles.get(), portChannels.data(), portChannels.size(),
            cudaMemcpyHostToDevice);

  // Keep registered memory references to prevent deallocation
  std::transform(remoteRegMemories.begin(), remoteRegMemories.end(),
                 std::back_inserter(ctx->registeredMemories),
                 [](const auto& fut) { return fut.get(); });
  ctx->registeredMemories.push_back(inputBufRegMem);
  ctx->registeredMemories.push_back(outputBufRegMem);

  return ctx;
}

AlgorithmCtxKey AlltoallvFullmesh::generateAlltoallvContextKey(
    const void* input, void* output, size_t inputSize, size_t outputSize,
    [[maybe_unused]] DataType dtype) {
  return {(void*)input, output, inputSize, outputSize, 0};
}

#undef ALLTOALLV_WARP_SIZE

}  // namespace collective
}  // namespace mscclpp
