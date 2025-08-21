// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "allreduce.hpp"
#include "debug.h"

using AllreduceFunc = std::function<cudaError_t(
    const void*, void*, void*, mscclpp::DeviceHandle<mscclpp::MemoryChannel>*,
    mscclpp::DeviceHandle<mscclpp::MemoryChannel>*, mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
    mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int, int, int, size_t, cudaStream_t,
    uint32_t*, uint32_t*, uint32_t*, uint32_t)>;

namespace {

template <Op OpType, typename T>
struct AllpairAdapter {
  static cudaError_t call(const void* buff, void* scratch, void* resultBuff,
                          mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                          mscclpp::DeviceHandle<mscclpp::MemoryChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t channelInOffset, size_t,
                          size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize, size_t nelems,
                          cudaStream_t stream, uint32_t* deviceFlag7, uint32_t* deviceFlag28, uint32_t*,
                          uint32_t numScratchBuff) {
    if (sizeof(T) * nelems < worldSize * sizeof(int)) {
      int nBlocks = worldSize - 1;
      int nThreadsPerBlock = 32;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag7, numScratchBuff);
    } else if (sizeof(T) * nelems <= (1 << 14)) {
      int nBlocks = (worldSize - 1) * 4;
      int nThreadsPerBlock = 512;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag28, numScratchBuff);
    }
    return cudaGetLastError();
  }
};

template <template <Op, typename> class Adapter>
AllreduceFunc dispatch(ncclRedOp_t op, ncclDataType_t dtype) {
  Op reduceOp = getReduceOp(op);
  AllreduceFunc allreduceFunc;
  if (reduceOp == SUM) {
    if (dtype == ncclFloat16) {
      allreduceFunc = Adapter<SUM, half>::call;
    } else if (dtype == ncclFloat32) {
      allreduceFunc = Adapter<SUM, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == ncclBfloat16) {
      allreduceFunc = Adapter<SUM, __bfloat16>::call;
#endif
    } else if (dtype == ncclInt32 || dtype == ncclUint32) {
      allreduceFunc = Adapter<SUM, int>::call;
    } else {
      return nullptr;
    }
  } else if (reduceOp == MIN) {
    if (dtype == ncclFloat16) {
      allreduceFunc = Adapter<MIN, half>::call;
    } else if (dtype == ncclFloat32) {
      allreduceFunc = Adapter<MIN, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == ncclBfloat16) {
      allreduceFunc = Adapter<MIN, __bfloat16>::call;
#endif
    } else if (dtype == ncclInt32 || dtype == ncclUint32) {
      allreduceFunc = Adapter<MIN, int>::call;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}
}  // namespace

enum Op getReduceOp(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return SUM;
    case ncclMin:
      return MIN;
    default:
      WARN("op is invalid, op: %d", op);
      throw mscclpp::Error("Invalid operation", mscclpp::ErrorCode::InternalError);
  }
}

ncclResult_t AllreduceAllpair::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                   void* output, size_t count, [[maybe_unused]] ncclDataType_t dtype,
                                                   cudaStream_t stream,
                                                   std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  const size_t bytes = count * ncclTypeSize(dtype);
  const int worldSize = ctx->workSize;
  ncclRedOp_t op = *static_cast<ncclRedOp_t*>(extras.at("op").get());
  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d", op, dtype);
    return ncclInvalidArgument;
  }
  cudaError_t error = allreduce(input, ctx->scratchBuffer.get(), output, ctx->memoryChannelDeviceHandles.get(), nullptr,
                                nullptr, nullptr, 0, 0, 0, ctx->rank, ctx->nRanksPerNode, ctx->workSize, count, stream,
                                nullptr, nullptr, nullptr, 0U);
  if (error != cudaSuccess) {
    WARN("AllreduceAllpair failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceAllpair::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, const void*, void* output, size_t, ncclDataType_t) {
      return nullptr;
    }
mscclpp::AlgorithmCtxKey AllreduceAllpair::generateAllreduceContextKey(const void*, void*, size_t, ncclDataType_t) {
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

void AllreduceAllpair::registerAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
  auto self = shared_from_this();
  mscclpp::Algorithm allgatherAlgo(
      comm, "allreduce",
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t count,
             ncclDataType_t dtype, cudaStream_t stream,
             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
        return self->allreduceKernelFunc(ctx, input, output, count, dtype, stream, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
             ncclDataType_t dtype) { return self->initAllreduceContext(comm, input, output, count, dtype); },
      [self](const void* input, void* output, size_t count, ncclDataType_t dtype) {
        return self->generateAllreduceContextKey(input, output, count, dtype);
      });
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("allreduce", "default_allreduce_allpair", allgatherAlgo);
}