// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "allgather.hpp"
#include "debug.h"

AllgatherAlgo6::AllgatherAlgo6() : disableChannelCache_(false) {
  if (mscclpp::env()->disableChannelCache) {
    disableChannelCache_ = true;
  }
}

ncclResult_t AllgatherAlgo6::allgatherKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                 void* output, size_t count, ncclDataType_t dtype, cudaStream_t stream,
                                                 std::unordered_map<std::string, std::shared_ptr<void>>&) {
  int nBlocks = 28;
  const size_t bytes = count * ncclTypeSize(dtype);
  const size_t nElem = bytes / sizeof(int);
  int rank = ctx->rank;
  if (bytes <= 32 * (1 << 20)) {
    if (nElem <= 4096) {
      nBlocks = 7;
    } else if (nElem <= 32768) {
      nBlocks = 14;
    } else if (nElem >= 2097152) {
      nBlocks = 35;
    }
  } else {
    nBlocks = 35;
  }

  size_t channelOutOffset = reinterpret_cast<size_t>(ctx->extras["channel_out_offset"].get());
  if ((char*)input == (char*)output + rank * bytes) {
    allgather6<false><<<nBlocks, 1024, 0, stream>>>((void*)input, ctx->memoryChannelDeviceHandles.get(),
                                                    channelOutOffset, ctx->rank, ctx->workSize, ctx->nRanksPerNode,
                                                    nElem);
  } else {
    allgather6<true><<<nBlocks, 1024, 0, stream>>>((void*)input, ctx->memoryChannelDeviceHandles.get(),
                                                   channelOutOffset, ctx->rank, ctx->workSize, ctx->nRanksPerNode,
                                                   nElem);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherAlgo6 failed with error %d", err);
    return ncclInternalError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllgatherAlgo6::initAllgatherContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                                            const void*, void* output, size_t count,
                                                                            ncclDataType_t dtype) {
  constexpr int nChannelsPerConnection = 35;

  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup connections
  ctx->connections = std::move(setupConnections(comm));
  // setup semaphores
  ctx->memorySemaphores = std::move(setupMemorySemaphores(comm, ctx->connections, nChannelsPerConnection));

  size_t bytes = count * ncclTypeSize(dtype);
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  if (disableChannelCache_) {
    channelOutOffset = 0;
    recvBytes = bytes;
    recvBasePtr = (CUdeviceptr)output;
  }
  ctx->extras.insert({"channel_out_offset", std::make_shared<size_t>(channelOutOffset)});

  // register the memory for the broadcast operation
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)recvBasePtr, recvBytes, mscclpp::Transport::CudaIpc);
  std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, localMemory);
  ctx->memoryChannels = std::move(setupMemoryChannels(ctx->connections, ctx->memorySemaphores, remoteMemories,
                                                      localMemory, nChannelsPerConnection));
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);

  return ctx;
}

mscclpp::AlgorithmCtxKey AllgatherAlgo6::generateAllgatherContextKey(const void*, void* output, size_t,
                                                                     ncclDataType_t) {
  static int tag = 0;
  if (disableChannelCache_) {
    // always return a new key if channel cache is disabled
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, tag++};
  }
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

void AllgatherAlgo6::registerAllgatherAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
  mscclpp::Algorithm allgatherAlgo(
      comm, "allgather",
      [this](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count,
             ncclDataType_t dtype, cudaStream_t stream,
             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
        return allgatherKernelFunc(ctx, input, output, count, dtype, stream, extras);
      },
      [this](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
             ncclDataType_t dtype) { return initAllgatherContext(comm, input, output, count, dtype); },
      [this](const void* input, void* output, size_t count, ncclDataType_t dtype) {
        return generateAllgatherContextKey(input, output, count, dtype);
      });
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("allgather", "default_allgather6", allgatherAlgo);
}
