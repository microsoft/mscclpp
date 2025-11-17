// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "broadcast.hpp"
#include "datatype_conversion.hpp"

void BroadcastAlgo6::initialize(std::shared_ptr<mscclpp::Communicator> comm,
                                std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  this->conns_ = setupConnections(comm);
  this->scratchBuffer_ = std::static_pointer_cast<char>(extras.at("scratch"));
  this->scratchMemSize_ = *(size_t*)(extras.at("scratch_size").get());
}

ncclResult_t BroadcastAlgo6::broadcastKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                 void* output, size_t count, mscclpp::DataType dtype,
                                                 cudaStream_t stream,
                                                 std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  int root = *(int*)extras.at("root").get();
  const size_t elemSize = getDataTypeSize(dtype);
  cudaError_t err;
  if (input == output) {
    err = broadcast<false>((int*)input, (int*)this->scratchBuffer_.get(), (int*)output,
                           ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                           count * elemSize / sizeof(int), stream);
  } else {
    err = broadcast<true>((int*)input, (int*)this->scratchBuffer_.get(), (int*)output,
                          ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root, ctx->workSize,
                          count * elemSize / sizeof(int), stream);
  }
  if (err != cudaSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> BroadcastAlgo6::initBroadcastContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                                            const void*, void* output, size_t,
                                                                            mscclpp::DataType) {
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
      comm->registerMemory(this->scratchBuffer_.get(), scratchMemSize_, mscclpp::Transport::CudaIpc);
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

mscclpp::AlgorithmCtxKey BroadcastAlgo6::generateBroadcastContextKey(const void*, void*, size_t, mscclpp::DataType) {
  // always use same context
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

mscclpp::Algorithm BroadcastAlgo6::build() {
  auto self = std::make_shared<BroadcastAlgo6>();
  mscclpp::Algorithm broadcastAlgo(
      "default_broadcast6", "broadcast",
      [self](std::shared_ptr<mscclpp::Communicator> comm,
             std::unordered_map<std::string, std::shared_ptr<void>>& extras) { self->initialize(comm, extras); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count,
             mscclpp::DataType dtype, cudaStream_t stream,
             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
        return self->broadcastKernelFunc(ctx, input, output, count, dtype, stream, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
             mscclpp::DataType dtype) { return self->initBroadcastContext(comm, input, output, count, dtype); },
      [self](const void* input, void* output, size_t count, mscclpp::DataType dtype) {
        return self->generateBroadcastContextKey(input, output, count, dtype);
      });
  return broadcastAlgo;
}
