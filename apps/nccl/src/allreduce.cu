// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nccl.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "allreduce.hpp"
#include "debug.h"

using AllreduceFunc =
    std::function<cudaError_t(const void*, void*, void*, void*, void*, mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                              mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int, int, int,
                              size_t, cudaStream_t, uint32_t*, uint32_t*, uint32_t*, uint32_t)>;

namespace {

template <Op OpType, typename T>
struct AllpairAdapter {
  static cudaError_t call(const void* buff, void* scratch, void* resultBuff, void* memoryChannels, void*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t channelInOffset, size_t,
                          size_t scratchBufferSize, int rank, int nRanksPerNode, int worldSize, size_t nelems,
                          cudaStream_t stream, uint32_t* deviceFlag7, uint32_t* deviceFlag28, uint32_t* deviceFlag56,
                          uint32_t numScratchBuff) {
    using ChannelType = mscclpp::DeviceHandle<mscclpp::MemoryChannel>;
    if (sizeof(T) * nelems < worldSize * sizeof(int)) {
      int nBlocks = worldSize - 1;
      int nThreadsPerBlock = 32;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag7, numScratchBuff);
    } else if (sizeof(T) * nelems <= (1 << 14)) {
      int nBlocks = (worldSize - 1) * 4;
      int nThreadsPerBlock = 512;
      allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag28, numScratchBuff);
    } else if (sizeof(T) * nelems <= (1 << 20)) {
      int nBlocks = (nRanksPerNode - 1) * 4;
      int nThreadsPerBlock = 1024;
      uint32_t* deviceFlag = deviceFlag28;
      if (nelems >= 8192) {
        nBlocks = (worldSize - 1) * 8;
        nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
        deviceFlag = deviceFlag56;
      }
#if defined(ENABLE_NPKIT)
      size_t NpkitSharedMemSize = NPKIT_SHM_NUM_EVENTS * sizeof(NpKitEvent);
      allreduce7<OpType><<<nBlocks, nThreadsPerBlock, NpkitSharedMemSize, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag, numScratchBuff, NpKit::GetGpuEventCollectContexts(),
          NpKit::GetCpuTimestamp());
#else
      allreduce7<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
          (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
          nRanksPerNode, worldSize, nelems, deviceFlag, numScratchBuff);
#endif
    }
    return cudaGetLastError();
  }
};

template <Op OpType, typename T>
struct NvlsAdapter {
  static cudaError_t call(const void*, void*, void*, void* memoryChannels, void*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels, size_t channelInOffset,
                          size_t channelOutOffset, size_t, int rank, int nRanksPerNode, int, size_t nelems,
                          cudaStream_t stream, uint32_t*, uint32_t*, uint32_t*, uint32_t) {
    using ChannelType = mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>;
    int nBlocks = nRanksPerNode;
    int nThreadsPerBlock = 1024;
    // printf(
    //     "in call nvlsChannels %p, nvlsOutChannels %p, channelInOffset %ld, channelOutOffset %ld, nelems %ld, rank %d, "
    //     "nRanksPerNode %d\n",
    //     nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset, nelems, rank, nRanksPerNode);
    // allreduce9<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>((ChannelType*)memoryChannels, nvlsChannels, nvlsOutChannels,
    //                                                         channelInOffset, channelOutOffset, nelems * sizeof(T), rank,
    //                                                         nRanksPerNode);
    return cudaGetLastError();
  }
};

template <template <Op, typename> class Adapter>
AllreduceFunc dispatch(ncclRedOp_t op, ncclDataType_t dtype) {
  Op reduceOp = getReduceOp(op);
  if (reduceOp == SUM) {
    if (dtype == ncclFloat16) {
      return Adapter<SUM, half>::call;
    } else if (dtype == ncclFloat32) {
      return Adapter<SUM, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == ncclBfloat16) {
      return Adapter<SUM, __bfloat16>::call;
#endif
    } else if (dtype == ncclInt32 || dtype == ncclUint32) {
      return Adapter<SUM, int>::call;
    } else {
      return nullptr;
    }
  } else if (reduceOp == MIN) {
    if (dtype == ncclFloat16) {
      return Adapter<MIN, half>::call;
    } else if (dtype == ncclFloat32) {
      return Adapter<MIN, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == ncclBfloat16) {
      return Adapter<MIN, __bfloat16>::call;
#endif
    } else if (dtype == ncclInt32 || dtype == ncclUint32) {
      return Adapter<MIN, int>::call;
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

AllreducePacket::AllreducePacket()
    : scratchBuffer_(mscclpp::GpuBuffer<char>(1 << 25)),
      deviceFlag7_(mscclpp::detail::gpuCallocShared<uint32_t>(7)),
      deviceFlag28_(mscclpp::detail::gpuCallocShared<uint32_t>(28)),
      deviceFlag56_(mscclpp::detail::gpuCallocShared<uint32_t>(56)),
      ctx_(nullptr) {
  std::vector<uint32_t> initFlag(56);
  for (int i = 0; i < 56; ++i) {
    initFlag[i] = 1;
  }
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag7_.get(), initFlag.data(), 7, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag28_.get(), initFlag.data(), 28, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag56_.get(), initFlag.data(), 56, cudaMemcpyHostToDevice);
}

ncclResult_t AllreducePacket::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                   void* output, size_t count, [[maybe_unused]] ncclDataType_t dtype,
                                                   cudaStream_t stream,
                                                   std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  const size_t bytes = count * ncclTypeSize(dtype);
  const int worldSize = ctx->workSize;
  ncclRedOp_t op = *static_cast<ncclRedOp_t*>(extras.at("op").get());

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d", op, dtype);
    return ncclInvalidArgument;
  }
  cudaError_t error = allreduce(input, this->scratchBuffer_.data(), output, ctx->memoryChannelDeviceHandles.get(),
                                nullptr, nullptr, nullptr, channelInOffset, 0, this->scratchBuffer_.bytes(), ctx->rank,
                                ctx->nRanksPerNode, ctx->workSize, count, stream, deviceFlag7_.get(),
                                deviceFlag28_.get(), deviceFlag56_.get(), this->nSegmentsForScratchBuffer_);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreducePacket::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* , size_t, ncclDataType_t) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  const int nChannelsPerConnection = 56;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  if (this->ctx_ == nullptr) {
    // setup connections
    ctx->connections = std::move(setupConnections(comm));
    // setup semaphores
    ctx->memorySemaphores = std::move(setupMemorySemaphores(comm, ctx->connections, nChannelsPerConnection));
    // setup registered memories
    mscclpp::RegisteredMemory scratchMemory =
        comm->registerMemory(this->scratchBuffer_.data(), this->scratchBuffer_.bytes(), mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, scratchMemory);
    ctx->registeredMemories = std::move(remoteMemories);
    ctx->registeredMemories.push_back(scratchMemory);
  } else {
    ctx->connections = ctx_->connections;
    ctx->memorySemaphores = ctx_->memorySemaphores;
    ctx->registeredMemories = ctx_->registeredMemories;
    ctx->registeredMemories.pop_back(); // remove the local memory from previous context
  }

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)sendBasePtr, sendBytes, mscclpp::Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = std::move(setupMemoryChannels(ctx->connections, ctx->memorySemaphores, ctx->registeredMemories,
                                                      localMemory, nChannelsPerConnection));
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);

  this->ctx_ = ctx;

  return ctx;
}

mscclpp::AlgorithmCtxKey AllreducePacket::generateAllreduceContextKey(const void* input, void*, size_t,
                                                                       ncclDataType_t) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

void AllreducePacket::registerAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
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
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("allreduce", "default_allreduce_packet", allgatherAlgo);
}

AllreduceNvls::AllreduceNvls(std::shared_ptr<mscclpp::Communicator> comm) {
  nSwitchChannel_ = 8;
  this->conns_ = std::move(setupConnections(comm));
  // setup semaphores
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores =
      std::move(setupMemorySemaphores(comm, this->conns_, nSwitchChannel_));
  // setup base memory channels
  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, memorySemaphores, nSwitchChannel_);
  this->memoryChannelsDeviceHandle_ = setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
}

ncclResult_t AllreduceNvls::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                void* output, size_t count, ncclDataType_t dtype, cudaStream_t stream,
                                                std::unordered_map<std::string, std::shared_ptr<void>>&) {
  AllreduceFunc allreduce = dispatch<NvlsAdapter>(ncclSum, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d", dtype);
    return ncclInvalidArgument;
  }
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels = ctx->switchChannelDeviceHandles.get();
  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsOutChannels = ctx->switchChannelDeviceHandles.get();
  if (input != output) {
    nvlsOutChannels = nvlsOutChannels + nSwitchChannel_;
  }
  // printf(
  //     "input %p, output %p, sendBasePtr %p, recvBasePtr %p nvlsinChannels %p, nvlsOutChannels %p, channelInOffset %ld, "
  //     "channelOutOffset %ld, nelems "
  //     "%ld, rank %d, "
  //     "nRanksPerNode %d\n",
  //     input, output, sendBasePtr, recvBasePtr, nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset, count,
  //     ctx->rank, ctx->nRanksPerNode);
  cudaError_t error = allreduce(nullptr, nullptr, nullptr, this->memoryChannelsDeviceHandle_.get(), nullptr,
                                nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset, 0, ctx->rank,
                                ctx->nRanksPerNode, ctx->workSize, count, stream, nullptr, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

mscclpp::AlgorithmCtxKey AllreduceNvls::generateAllreduceContextKey(const void* input, void* output, size_t,
                                                                    ncclDataType_t) {
  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, (void*)recvBasePtr, sendBytes, recvBytes, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceNvls::initAllreduceContext(std::shared_ptr<mscclpp::Communicator> comm,
                                                                           const void* input, void* output,
                                                                           size_t, ncclDataType_t) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  size_t sendBytes, recvBytes;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // setup channels
  ctx->nvlsConnections = std::move(setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannel_));
  ctx->switchChannels =
      std::move(setupNvlsChannels(ctx->nvlsConnections, (void*)sendBasePtr, sendBytes, nSwitchChannel_));
  if (input != output) {
    auto nvlsOutConnections = std::move(setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannel_));
    std::vector<mscclpp::SwitchChannel> outChannels =
        setupNvlsChannels(nvlsOutConnections, (void*)recvBasePtr, recvBytes, nSwitchChannel_);
    ctx->nvlsConnections.insert(ctx->nvlsConnections.end(), nvlsOutConnections.begin(), nvlsOutConnections.end());
    ctx->switchChannels.insert(ctx->switchChannels.end(), outChannels.begin(), outChannels.end());
  }

  // ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

void AllreduceNvls::registerAlgorithm(std::shared_ptr<mscclpp::Communicator> comm) {
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
  mscclpp::AlgorithmFactory::getInstance()->registerAlgorithm("allreduce", "default_allreduce_nvls", allgatherAlgo);
}
