// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <collective_utils.hpp>
#include <type_traits>

#include "allreduce/allreduce_allpair_packet.hpp"
#include "allreduce/common.hpp"
#include "collective_utils.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {

template <ReduceOp OpType, typename T, typename AccumT = T>
__global__ void allreduceAllPairs(T* buff, T* scratch, T* resultBuff, DeviceHandle<MemoryChannel>* memoryChannels,
                                  size_t channelDataOffset, size_t scratchBufferSize, int rank, int nRanksPerIpcDomain,
                                  int worldSize, size_t nelems, uint32_t numScratchBuff, void* flags,
                                  uint32_t flagSize) {
  if (sizeof(T) == 2 || sizeof(T) == 1) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerIpcDomain - 1;

  uint32_t flag = ((uint32_t*)flags)[blockIdx.x];
  size_t scratchBaseOffset = (flag % numScratchBuff) ? (scratchBufferSize / numScratchBuff) : 0;
  size_t channelScratchOffset = scratchBaseOffset;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  const int warpId = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int nWarpsPerBlock = blockDim.x / WARP_SIZE;
  // Assign one warp in every block to each peer. Each peer warp sends the
  // same block-owned stripe, so nBlocks only partitions data and no longer
  // needs to be grouped by nPeers.
  if (warpId < nPeers) {
    memoryChannels[warpId].putPackets<LL8Packet>(scratchOffset, channelDataOffset, nelems * sizeof(uint32_t),
                                                 lane + blockIdx.x * WARP_SIZE, gridDim.x * WARP_SIZE, flag);
  }
  // Safe for in-place allreduce: all peer warps must finish reading src for
  // this block's stripe before any warp writes reduced data back to dst/src.
  __syncthreads();

  // Split the same sent stream across all warps for reduction. warpId selects
  // which strided subset to reduce while lane preserves coalesced packet reads.
  for (size_t idx = lane + blockIdx.x * WARP_SIZE + warpId * WARP_SIZE * gridDim.x; idx < nelems;
       idx += nWarpsPerBlock * WARP_SIZE * gridDim.x) {
    uint32_t data = src[idx];
    using AccRaw = std::conditional_t<std::is_same_v<T, AccumT>, uint32_t,
                                      mscclpp::VectorType<AccumT, sizeof(uint32_t) / sizeof(T)>>;
    AccRaw acc = mscclpp::upcastVector<T, AccumT, AccRaw>(data);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      LL8Packet* dstPkt = (LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      acc = mscclpp::calVectorAccum<T, AccumT, OpType, AccRaw>(acc, val);
    }
    dst[idx] = mscclpp::downcastVector<T, AccumT, uint32_t>(acc);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ((uint32_t*)flags)[blockIdx.x] = flag + 1;
  }
  if (tid >= gridDim.x && tid < flagSize / sizeof(uint32_t)) {
    ((uint32_t*)flags)[tid] = flag + 1;
  }
}

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize, int nRanksPerIpcDomain) {
  if (inputSize < nRanksPerIpcDomain * sizeof(int)) {
    return {nRanksPerIpcDomain - 1, (nRanksPerIpcDomain - 1) * WARP_SIZE};
  }
  return {(nRanksPerIpcDomain - 1) * 4, 512};
}

template <ReduceOp OpType, typename T, typename AccumT = T>
struct AllpairAdapter {
  static cudaError_t call(const void* buff, void* scratch, void* resultBuff, void* memoryChannels, void*,
                          DeviceHandle<SwitchChannel>*, DeviceHandle<SwitchChannel>*, size_t channelInOffset, size_t,
                          size_t scratchBufferSize, int rank, int nRanksPerIpcDomain, int worldSize, size_t inputSize,
                          cudaStream_t stream, void* flags, uint32_t flagSize, uint32_t numScratchBuff, int nBlocks = 0,
                          int nThreadsPerBlock = 0) {
    using ChannelType = DeviceHandle<MemoryChannel>;
    const size_t nelems = inputSize / sizeof(T);
    allreduceAllPairs<OpType, T, AccumT><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
        nRanksPerIpcDomain, worldSize, nelems, numScratchBuff, flags, flagSize);
    return cudaGetLastError();
  }
};

void AllreduceAllpairPacket::initialize(std::shared_ptr<Communicator> comm) {
  conns_ = setupConnections(comm);
  memorySemaphores_ = setupMemorySemaphores(comm, conns_, maxBlockNum_);
  RegisteredMemory scratchMemory = comm->registerMemory(scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  registeredMemories_ = setupRemoteMemories(comm, comm->bootstrap()->getRank(), scratchMemory);
  registeredMemories_.push_back(scratchMemory);
}

CommResult AllreduceAllpairPacket::allreduceKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                                       size_t inputSize, [[maybe_unused]] DataType dtype, ReduceOp op,
                                                       cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                                       const std::unordered_map<std::string, uintptr_t>&,
                                                       DataType accumDtype) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  if (algoCtx->worldSize != algoCtx->nRanksPerIpcDomain) {
    WARN(ALGO,
         "AllreduceAllpairPacket requires worldSize to match nRanksPerIpcDomain, got worldSize=", algoCtx->worldSize,
         ", nRanksPerIpcDomain=", algoCtx->nRanksPerIpcDomain);
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> blockAndThreadNum{nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize, algoCtx->nRanksPerIpcDomain);
  }
  if (blockAndThreadNum.first > maxBlockNum_) {
    WARN(ALGO, "Requested block number ", blockAndThreadNum.first, " exceeds the maximum supported block number ",
         maxBlockNum_, ".");
    return CommResult::CommInvalidArgument;
  }
  const int nPeers = algoCtx->nRanksPerIpcDomain - 1;
  // The kernel maps peer sends by warpId, so every peer needs a full warp.
  if (blockAndThreadNum.second % WARP_SIZE != 0 || blockAndThreadNum.second / WARP_SIZE < nPeers) {
    WARN(ALGO,
         "Allpair packet requires at least one full warp per peer, but got nThreadsPerBlock=", blockAndThreadNum.second,
         " and nPeers=", nPeers, ".");
    return CommResult::CommInvalidArgument;
  }
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype, accumDtype);
  if (!allreduce) {
    WARN(ALGO, "Unsupported operation or data type for allreduce: op=", static_cast<int>(op),
         ", dtype=", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output, algoCtx->memoryChannelDeviceHandles.get(), nullptr, nullptr,
                nullptr, channelInOffset, 0, this->scratchBufferSize_, algoCtx->rank, algoCtx->nRanksPerIpcDomain,
                algoCtx->worldSize, inputSize, stream, (void*)flagBuffer_, (uint32_t)flagBufferSize_,
                this->nSegmentsForScratchBuffer_, blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN(ALGO, "AllreducePacket failed with error: ", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

std::shared_ptr<void> AllreduceAllpairPacket::initAllreduceContext(std::shared_ptr<Communicator> comm,
                                                                   const void* input, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  const int nChannelsPerConnection = maxBlockNum_;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->worldSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerIpcDomain = comm->bootstrap()->getNranksPerIpcDomain();
  ctx->memorySemaphores = this->memorySemaphores_;
  ctx->registeredMemories = this->registeredMemories_;
  ctx->registeredMemories.pop_back();  // remove the local memory from previous context

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  RegisteredMemory localMemory = comm->registerMemory((void*)sendBasePtr, sendBytes, Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
                                            nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);
  return ctx;
}

AlgorithmCtxKey AllreduceAllpairPacket::generateAllreduceContextKey(const void* input, void*, size_t, DataType, bool) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<Algorithm> AllreduceAllpairPacket::build() {
  auto self = std::make_shared<AllreduceAllpairPacket>(reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_,
                                                       flagBuffer_, flagBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_allpair_packet", "allreduce",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras, DataType accumDtype) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, op, stream, nBlocks, nThreadsPerBlock,
                                         extras, accumDtype);
      },
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize, DataType dtype,
             bool symmetricMemory) {
        return self->generateAllreduceContextKey(input, output, inputSize, dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp
