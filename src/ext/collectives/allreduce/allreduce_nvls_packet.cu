// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allreduce/allreduce_nvls_packet.hpp"
#include "allreduce/common.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

__device__ uint32_t deviceFlag = 1;
template <ReduceOp OpType, typename T, bool flagPerBlock = false>
__global__ void __launch_bounds__(1024, 1)
    allreduceNvlsPacket([[maybe_unused]] const T* input, [[maybe_unused]] T* scratch, [[maybe_unused]] T* output,
                        [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
                        [[maybe_unused]] size_t nelems, [[maybe_unused]] size_t scratchBufferSize,
                        [[maybe_unused]] int rank, [[maybe_unused]] int worldSize, [[maybe_unused]] void* flags) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t flag = 0;
  if constexpr (flagPerBlock) {
    flag = ((uint32_t*)flags)[blockIdx.x];
  } else {
    flag = deviceFlag;
    __syncthreads();
    if (threadIdx.x == 0) {
      ((LL8Packet*)flags)[blockIdx.x].write(0, flag);
    }
  }

  size_t scratchBaseOffset = (flag % 2) ? scratchBufferSize / 2 : 0;
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t nPktPerRank = nelems / worldSize / (sizeof(mscclpp::LL8Packet::Payload) / sizeof(T));
  mscclpp::LL8Packet* multiPkt =
      (mscclpp::LL8Packet*)((char*)multicast->mcPtr + scratchBaseOffset) + rank * worldSize * nPktPerRank;
  uint* src = (uint*)(input);
  uint* dst = (uint*)(output);
  mscclpp::LL8Packet* scratchPkt = (mscclpp::LL8Packet*)((char*)scratch + scratchBaseOffset);
  for (uint32_t i = tid; i < nPktPerRank * worldSize; i += blockDim.x * gridDim.x) {
    mscclpp::LL8Packet pkt(src[i], flag);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(*(mscclpp::f32x2*)(&pkt), multiPkt + i);
  }
  for (uint32_t i = tid; i < nPktPerRank * worldSize; i += blockDim.x * gridDim.x) {
    uint data = src[i];
    for (int peer = 0; peer < worldSize; peer++) {
      if (peer == rank) {
        continue;
      }
      uint val = scratchPkt[peer * worldSize * nPktPerRank + i].read(flag);
      data = cal_vectors<T, OpType>(data, val);
    }
    dst[i] = data;
  }
  if constexpr (flagPerBlock) {
    __syncthreads();
    if (threadIdx.x == 0) {
      ((uint32_t*)flags)[blockIdx.x] = flag + 1;
    }
  } else {
    if (blockIdx.x == 0 && threadIdx.x < gridDim.x) {
      ((LL8Packet*)flags)[threadIdx.x].read(flag, -1);
    }
    if (blockIdx.x == 0) {
      __syncthreads();
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      deviceFlag++;
    }
  }
#endif
}

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize) {
  int blockNum = 8;
  int threadNum = 1024;
  if (inputSize <= (1 << 13)) {
    blockNum = 4;
    threadNum = 512;
  }
  return {blockNum, threadNum};
}

template <ReduceOp OpType, typename T>
struct AllreduceNvlsPacketAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void*, void*,
                          DeviceHandle<SwitchChannel>* nvlsChannels, DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t scratchBufferSize, int rank, int, int worldSize, size_t inputSize, cudaStream_t stream,
                          void* flags, uint32_t, int nBlocks, int nThreadsPerBlock) {
    if (nBlocks == 4 || nBlocks == 8) {
      allreduceNvlsPacket<OpType, T, true>
          <<<nBlocks, nThreadsPerBlock, 0, stream>>>((const T*)input, (T*)scratch, (T*)output, nvlsChannels,
                                                     inputSize / sizeof(T), scratchBufferSize, rank, worldSize, flags);
    } else {
      allreduceNvlsPacket<OpType, T>
          <<<nBlocks, nThreadsPerBlock, 0, stream>>>((const T*)input, (T*)scratch, (T*)output, nvlsChannels,
                                                     inputSize / sizeof(T), scratchBufferSize, rank, worldSize, flags);
    }
    return cudaGetLastError();
  }
};

void AllreduceNvlsPacket::initialize(std::shared_ptr<Communicator>) {
  std::vector<uint32_t> flags(8, 1);
  flags_ = detail::gpuCallocShared<LL8Packet>(16);
  flags4_ = detail::gpuCallocShared<uint32_t>(4);
  flags8_ = detail::gpuCallocShared<uint32_t>(8);
  gpuMemcpy<uint32_t>(flags4_.get(), flags.data(), 4, cudaMemcpyHostToDevice);
  gpuMemcpy<uint32_t>(flags8_.get(), flags.data(), 8, cudaMemcpyHostToDevice);
}

AlgorithmCtxKey AllreduceNvlsPacket::generateAllreduceContextKey(const void*, void*, size_t, DataType) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<AlgorithmCtx> AllreduceNvlsPacket::initAllreduceContext(std::shared_ptr<Communicator> comm, const void*,
                                                                        void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  // setup channels
  int nSwitchChannels = 1;
  ctx->nvlsConnections = setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels);
  ctx->switchChannels =
      setupNvlsChannels(ctx->nvlsConnections, this->scratchBuffer_, this->scratchBufferSize_, nSwitchChannels);
  ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

CommResult AllreduceNvlsPacket::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                    void* output, size_t inputSize, mscclpp::DataType dtype,
                                                    ReduceOp op, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                                    const std::unordered_map<std::string, uintptr_t>&) {
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize);
  }
  if (blockAndThreadNum.first > maxBlockNum_) {
    WARN("Block number %d exceeds the maximum limit %d", blockAndThreadNum.first, maxBlockNum_);
    return CommResult::CommInvalidArgument;
  }
  AllreduceFunc allreduce = dispatch<AllreduceNvlsPacketAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  void* flags = this->flags_.get();
  if (blockAndThreadNum.first == 4) {
    flags = this->flags4_.get();
  } else if (blockAndThreadNum.first == 8) {
    flags = this->flags8_.get();
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output, nullptr, nullptr, ctx->switchChannelDeviceHandles.get(), nullptr,
                0, 0, this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize, inputSize, stream, flags,
                0, blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsPacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsPacket::build() {
  auto self = std::make_shared<AllreduceNvlsPacket>((uintptr_t)scratchBuffer_, scratchBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_packet", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, op, stream, nBlocks, nThreadsPerBlock,
                                         extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->generateAllreduceContextKey(input, output, inputSize, dtype); });
}
}  // namespace collective
}  // namespace mscclpp