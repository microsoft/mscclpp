// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>

#include "broadcast.hpp"

struct BroadcastAlgo0 {
  ncclResult_t broadcastKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void* input, void* output,
                                   size_t count, [[maybe_unused]] ncclDataType_t dtype, cudaStream_t stream) {
    int root = *(int*)ctx->extras.at("root").get();
    cudaError_t err;
    if (input == output) {
      err = broadcast<false>((int*)input, (int*)ctx->scratchBuffer.get(), (int*)output,
                             ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root,
                             ctx->workSize, count / sizeof(int), stream);
    } else {
      err = broadcast<true>((int*)input, (int*)ctx->scratchBuffer.get(), (int*)output,
                            ctx->memoryChannelDeviceHandles.get(), 0, ctx->rank, ctx->nRanksPerNode, root,
                            ctx->workSize, count / sizeof(int), stream);
    }
    if (err != cudaSuccess) {
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  std::shared_ptr<mscclpp::AlgorithmCtx> initBroadcastContext(std::shared_ptr<mscclpp::Communicator> comm) {
    auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
    ctx->rank = comm->bootstrap()->getRank();
    ctx->workSize = comm->bootstrap()->getNranks();
    ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

    // Initialize other members as needed, such as memory channels, scratch buffer, etc.

    return ctx;
  }

  mscclpp::AlgorithmCtxKey generateBroadcastContextKey(void*, void*, size_t, ncclDataType_t) {
    // always use same context
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
  }

  void registerBroadcastAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
    mscclpp::Algorithm broadcastAlgo(
        comm, "broadcast",
        [this](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, void* input, void* output, size_t count,
               ncclDataType_t dtype,
               cudaStream_t stream) { return broadcastKernelFunc(ctx, input, output, count, dtype, stream); },
        [this](std::shared_ptr<mscclpp::Communicator> comm) { return initBroadcastContext(comm); },
        [this](void* input, void* output, size_t count, ncclDataType_t dtype) {
          return generateBroadcastContextKey(input, output, count, dtype);
        });
    mscclpp::AlgorithmFactory::registerAlgorithm("broadcast", "nccl", broadcastAlgo);
  }
};
