// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "broadcast.hpp"

void BroadcastAlgo6::initialize(std::shared_ptr<mscclpp::Communicator> comm) { this->conns_ = setupConnections(comm); }

ncclResult_t BroadcastAlgo6::broadcastKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                 void* output, size_t count, [[maybe_unused]] ncclDataType_t dtype,
                                                 cudaStream_t stream,
                                                 std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  int root = *(int*)extras.at("root").get();
  cudaError_t err;
  if (input == output) {
    err = broadcast<false>((int*)input, (int*)this->scratchBuffer_.data(), (int*)output,
                           ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                           count * ncclTypeSize(dtype) / sizeof(int), stream);
  } else {
    err = broadcast<true>((int*)input, (int*)this->scratchBuffer_.data(), (int*)output,
                          ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                          count * ncclTypeSize(dtype) / sizeof(int), stream);
  }
  if (err != cudaSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> BroadcastAlgo6::initBroadcastContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                                            const void*, void* output, size_t,
                                                                            ncclDataType_t) {
  constexpr int nChannelsPerConnection = 8;

  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup semaphores
  ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  mscclpp::RegisteredMemory localScratchMemory =
      comm->registerMemory(this->scratchBuffer_.data(), scratchMemSize_, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, localScratchMemory);
  ctx->memoryChannels =
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories, localMemory, nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(localScratchMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey BroadcastAlgo6::generateBroadcastContextKey(const void*, void*, size_t, ncclDataType_t) {
  // always use same context
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

void BroadcastAlgo6::registerAlgorithm() {
  auto self = shared_from_this();
  mscclpp::Algorithm broadcastAlgo(
      "broadcast", [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count, int dtype,
             cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
        return self->broadcastKernelFunc(ctx, input, output, count, static_cast<ncclDataType_t>(dtype), stream, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype) {
        return self->initBroadcastContext(comm, input, output, count, static_cast<ncclDataType_t>(dtype));
      },
      [self](const void* input, void* output, size_t count, int dtype) {
        return self->generateBroadcastContextKey(input, output, count, static_cast<ncclDataType_t>(dtype));
      });
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("broadcast", "default_broadcast6", broadcastAlgo);
}
