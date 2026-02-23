// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "alltoallv/alltoallv_fullmesh.hpp"
#include "alltoallv/alltoallv_kernel.hpp"
#include "collective_utils.hpp"

#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
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
  std::vector<MemoryChannel> memoryChannels;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
};

AlltoallvFullmesh::~AlltoallvFullmesh() = default;

std::shared_ptr<Algorithm> AlltoallvFullmesh::build() {
  // Create a new shared_ptr that owns the object to keep it alive
  // This ensures the lambdas capturing 'self' have a valid object
  auto self = std::make_shared<AlltoallvFullmesh>();

  std::shared_ptr<Algorithm> alltoallvAlgo = std::make_shared<NativeAlgorithm>(
      "default_alltoallv_fullmesh", "alltoallv",  // name, collective (was swapped before)
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
  worldSize_ = comm->bootstrap()->getNranks();
  this->conns_ = setupConnections(comm);
}

CommResult AlltoallvFullmesh::alltoallvKernelFunc(
    const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
    size_t outputSize, [[maybe_unused]] DataType dtype, cudaStream_t stream,
    [[maybe_unused]] int nBlocks, int nThreadsPerBlock,
    const std::unordered_map<std::string, uintptr_t>& extras) {

  auto algoCtx = std::static_pointer_cast<AllToAllVContext>(ctx);
  int rank = algoCtx->rank;
  int worldSize = algoCtx->worldSize;

  // Extract send/recv counts and displacements from extras
  auto it_sendCounts = extras.find("sendCounts");
  auto it_sendDispls = extras.find("sendDispls");
  auto it_recvCounts = extras.find("recvCounts");
  auto it_recvDispls = extras.find("recvDispls");
  auto it_remoteRecvDispls = extras.find("remoteRecvDispls");

  if (it_sendCounts == extras.end() || it_sendDispls == extras.end() ||
      it_recvCounts == extras.end() || it_recvDispls == extras.end() ||
      it_remoteRecvDispls == extras.end()) {
    return CommResult::CommInternalError;
  }

  const size_t* d_sendCounts = reinterpret_cast<const size_t*>(it_sendCounts->second);
  const size_t* d_sendDispls = reinterpret_cast<const size_t*>(it_sendDispls->second);
  const size_t* d_recvCounts = reinterpret_cast<const size_t*>(it_recvCounts->second);
  const size_t* d_recvDispls = reinterpret_cast<const size_t*>(it_recvDispls->second);
  const size_t* d_remoteRecvDispls = reinterpret_cast<const size_t*>(it_remoteRecvDispls->second);

  // Use maximum threads (1024) for best bandwidth utilization
  const int threadsPerBlock = (nThreadsPerBlock > 0 && nThreadsPerBlock <= 1024) ? nThreadsPerBlock : 1024;

  // Size-adaptive algorithm selection based on message size and world size:
  // - Small messages (<1MB avg): use basic kernel (lower latency)
  // - Large messages (>=1MB avg) with small world (<=16): use pipelined kernel
  // - Large messages (>=1MB avg) with large world (>16): use ring kernel (avoids congestion)
  constexpr size_t SIZE_THRESHOLD = 1 << 20;  // 1MB
  constexpr int WORLD_SIZE_THRESHOLD = 16;
  size_t avgMsgSize = inputSize / worldSize;

  if (avgMsgSize < SIZE_THRESHOLD) {
    // Small messages: use basic kernel for lower latency
    alltoallvKernel<<<1, threadsPerBlock, 0, stream>>>(
        algoCtx->memoryChannelDeviceHandles.get(),
        rank, worldSize,
        input, output,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  } else if (worldSize > WORLD_SIZE_THRESHOLD) {
    // Large messages + large world: use ring kernel to avoid congestion
    alltoallvRingKernel<<<1, threadsPerBlock, 0, stream>>>(
        algoCtx->memoryChannelDeviceHandles.get(),
        rank, worldSize,
        input, output,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  } else {
    // Large messages + small world: use pipelined chunked kernel
    alltoallvPipelinedKernel<<<1, threadsPerBlock, 0, stream>>>(
        algoCtx->memoryChannelDeviceHandles.get(),
        rank, worldSize,
        input, output,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
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

  // Exchange output buffer registration with all peers (we write to peer's output buffer)
  std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, ctx->rank, outputBufRegMem);

  // Setup memory semaphores for synchronization (1 channel per peer)
  constexpr int nChannelsPerConnection = 1;
  ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);

  // Setup memory channels: we read from our input buffer, write to peer's output buffer
  ctx->memoryChannels = setupMemoryChannels(
      this->conns_,
      ctx->memorySemaphores,
      remoteOutputMemories,  // remote output buffers (where we write)
      inputBufRegMem,        // local input buffer (where we read from)
      nChannelsPerConnection);

  // Setup device handles
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // Keep registered memory references to prevent deallocation
  ctx->registeredMemories = std::move(remoteOutputMemories);
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
