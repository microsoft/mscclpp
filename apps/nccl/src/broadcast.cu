// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "broadcast.hpp"

ncclResult_t BroadcastAlgo0::broadcastKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                 void* output, size_t count, [[maybe_unused]] ncclDataType_t dtype,
                                                 cudaStream_t stream,
                                                 std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  int root = *(int*)extras.at("root").get();
  cudaError_t err;
  if (input == output) {
    err = broadcast<false>((int*)input, (int*)ctx->scratchBuffer.get(), (int*)output,
                           ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                           count * ncclTypeSize(dtype) / sizeof(int), stream);
  } else {
    err = broadcast<true>((int*)input, (int*)ctx->scratchBuffer.get(), (int*)output,
                          ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                          count * ncclTypeSize(dtype) / sizeof(int), stream);
  }
  if (err != cudaSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> BroadcastAlgo0::initBroadcastContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                                            const void*, void* output, size_t,
                                                                            ncclDataType_t) {
  constexpr int nChannelsPerConnection = 8;

  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup connections
  ctx->connections = std::move(setupConnections(comm));
  // setup semaphores
  ctx->memorySemaphores = std::move(setupMemorySemaphores(comm, ctx->connections, nChannelsPerConnection));

  // setup registered memories
  constexpr size_t scratchMemSize = 1 << 26;  // 64MB
  ctx->scratchBuffer = mscclpp::GpuBuffer(scratchMemSize).memory();

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  mscclpp::RegisteredMemory localScratchMemory =
      comm->registerMemory(ctx->scratchBuffer.get(), scratchMemSize, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, localScratchMemory);
  ctx->memoryChannels = std::move(setupMemoryChannels(ctx->connections, ctx->memorySemaphores, remoteMemories,
                                                      localMemory, nChannelsPerConnection));
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(localScratchMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey BroadcastAlgo0::generateBroadcastContextKey(const void*, void*, size_t, ncclDataType_t) {
  // always use same context
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

void BroadcastAlgo0::registerBroadcastAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
  mscclpp::Algorithm broadcastAlgo(
      comm, "broadcast",
      [this](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count,
             ncclDataType_t dtype, cudaStream_t stream,
             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
        return broadcastKernelFunc(ctx, input, output, count, dtype, stream, extras);
      },
      [this](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
             ncclDataType_t dtype) { return initBroadcastContext(comm, input, output, count, dtype); },
      [this](const void* input, void* output, size_t count, ncclDataType_t dtype) {
        return generateBroadcastContextKey(input, output, count, dtype);
      });
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("broadcast", "broadcast0", broadcastAlgo);
}
