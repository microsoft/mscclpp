// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <type_traits>

#include "allreduce/allreduce_rsag_zero_copy.hpp"
#include "allreduce/common.hpp"
#include "collective_utils.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {

__device__ mscclpp::DeviceSyncer globalSyncer;

// Zero-copy Reduce-Scatter + All-Gather (RSAG) allreduce.
//
// Unlike the standard RSAG which copies input into a scratch buffer first,
// this variant reads directly from peers' input buffers and writes reduced
// results directly to peers' output buffers — eliminating the need for a
// separate scratch buffer and reducing memory traffic.
//
// The algorithm runs in a single kernel with the following steps:
//
//   1. Barrier: Signal and wait on all peers to ensure input buffers are ready.
//
//   2. Reduce-Scatter: Each rank reads its assigned chunk from every peer's
//      input buffer (via CudaIpc remote memory handles), reduces all values
//      locally, then writes the reduced result to its own output buffer AND
//      directly to every peer's output buffer at the same offset.
//
//   3. Global sync + Barrier: A device-wide sync ensures all writes complete,
//      followed by a final signal/wait to guarantee all peers have finished
//      writing, making the full output buffer valid on every rank.
//
// This approach requires registering both input and output buffers as remote
// memories (2 * nPeers handles), but avoids scratch buffer allocation and
// the extra copy steps of the standard RSAG. ipcDomainNranks is accepted at
// runtime, which allows the same kernel to handle any NVLink-domain size
// (including Multi-Node NVLink fabrics up to NVL72).

template <ReduceOp OpType, typename T, typename AccumT = T>
__global__ void __launch_bounds__(1024, 1)
    allreduceRsAgZeroCopy(T* buff, T* scratch, T* resultBuff, DeviceHandle<BaseMemoryChannel>* memoryChannels,
                          DeviceHandle<SwitchChannel>* switchChannels, void* remoteMemories, int rank,
                          int ipcDomainNranks, int worldSize, size_t nelems) {
  int blockId = blockIdx.x;

  assert((uintptr_t)buff % sizeof(int4) == 0);
  assert((uintptr_t)resultBuff % sizeof(int4) == 0);

  const int NPeers = ipcDomainNranks - 1;
  constexpr uint32_t nelemsPerInt4 = sizeof(int4) / sizeof(T);
  const uint32_t outputRemoteBufferOffset = NPeers;
  uint32_t alignedNelems = ((nelems + ipcDomainNranks - 1) / ipcDomainNranks + nelemsPerInt4 - 1) / nelemsPerInt4 *
                           nelemsPerInt4 * ipcDomainNranks;
  uint32_t nelemsPerRank = alignedNelems / ipcDomainNranks;
  uint32_t nInt4PerRank = nelemsPerRank / nelemsPerInt4;
  uint32_t nInt4Total = (nelems + nelemsPerInt4 - 1) / nelemsPerInt4;

  int4* resultBuff4 = reinterpret_cast<int4*>((char*)resultBuff);
  int4* buff4 = reinterpret_cast<int4*>((char*)buff);
  DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal = memoryChannels + blockId * NPeers;

  uint32_t nInt4PerBlock = nInt4PerRank / gridDim.x;
  uint32_t remainderForBlock = nInt4PerRank % gridDim.x;
  uint32_t offset4 = blockId * nInt4PerBlock;
  if (blockId == (int)(gridDim.x - 1)) {
    nInt4PerBlock += remainderForBlock;
  }
  if (nInt4PerBlock == 0) return;

  if ((int)threadIdx.x < NPeers) {
    memoryChannelsLocal[threadIdx.x].relaxedSignal();
    memoryChannelsLocal[threadIdx.x].relaxedWait();
  }
  __syncthreads();
  // AccumInt4: when AccumT != T, use a wider accumulator type.
  // For AccumT == T, this is just int4 (no-op conversion).
  constexpr int nElemsPerInt4 = sizeof(int4) / sizeof(T);
  // When T == AccumT, stay with raw int4 to avoid type mismatch in identity path.
  using AccumVec = std::conditional_t<std::is_same_v<T, AccumT>, int4, mscclpp::VectorType<AccumT, nElemsPerInt4>>;
  for (uint32_t idx = threadIdx.x; idx < nInt4PerBlock; idx += blockDim.x) {
    uint32_t offset = idx + offset4 + rank * nInt4PerRank;
    if (offset >= nInt4Total) continue;
    int4 tmp_raw = buff4[offset];
    int4 data;
    AccumVec acc = mscclpp::upcastVector<T, AccumT, AccumVec>(tmp_raw);
    for (int i = 0; i < NPeers; i++) {
      int rankIdx = (rank + i + 1) % ipcDomainNranks;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      data = mscclpp::read<int4>(((void**)remoteMemories)[peerIdx], offset);
      acc = mscclpp::calVectorAccum<T, AccumT, OpType, AccumVec>(acc, data);
    }
    int4 tmp = mscclpp::downcastVector<T, AccumT, int4>(acc);
    for (int i = 0; i < NPeers; i++) {
      int rankIdx = (rank + i + 1) % ipcDomainNranks;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      mscclpp::write<int4>(((void**)remoteMemories)[outputRemoteBufferOffset + peerIdx], offset, tmp);
    }
    resultBuff4[offset] = tmp;
  }
  // Use device barrier gives better performance here.
  globalSyncer.sync(gridDim.x);
  if (blockIdx.x == 0 && (int)threadIdx.x < NPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
}

template <ReduceOp OpType, typename T, typename AccumT = T>
struct AllreduceRsAgZeroCopyAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel, DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t, int rank, int ipcDomainNranks, int worldSize, size_t inputSize, cudaStream_t stream,
                          void*, uint32_t, uint32_t, int nBlocks, int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = 64;
      if (inputSize >= (1 << 26)) {
        nBlocks = 128;
      }
    }
    allreduceRsAgZeroCopy<OpType, T, AccumT><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels, switchChannel, remoteMemories, rank,
        ipcDomainNranks, worldSize, nelems);
    return cudaGetLastError();
  }
};

void AllreduceRsAgZeroCopy::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 128;
  comm_ = comm;
  // setup semaphores
  this->semaphores_ = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, this->semaphores_, nChannelsPerConnection_);
  this->baseMemoryChannelHandles_ = setupBaseMemoryChannelDeviceHandles(baseChannels_);
}

CommResult AllreduceRsAgZeroCopy::allreduceKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                                      size_t inputSize, DataType dtype, ReduceOp op,
                                                      cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                                      const std::unordered_map<std::string, uintptr_t>&,
                                                      DataType accumDtype) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  AllreduceFunc allreduce = dispatch<AllreduceRsAgZeroCopyAdapter>(op, dtype, accumDtype);
  if (!allreduce) {
    WARN(ALGO, "Unsupported operation or data type for allreduce: op=", static_cast<int>(op),
         ", dtype=", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  if (numBlocksAndThreads.first > nChannelsPerConnection_) {
    WARN(ALGO, "Block number ", numBlocksAndThreads.first, " exceeds the maximum limit ", nChannelsPerConnection_);
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error =
      allreduce(input, nullptr, output, this->baseMemoryChannelHandles_.get(), algoCtx->remoteMemoryHandles.get(),
                nullptr, nullptr, 0, 0, 0, algoCtx->rank, algoCtx->ipcDomainNranks, algoCtx->workSize, inputSize,
                stream, nullptr, 0, 0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN(ALGO, "Allreduce kernel launch failed with error: ", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceRsAgZeroCopy::generateAllreduceContextKey(const void* inputBuffer, void* outputBuffer,
                                                                   size_t size, DataType, bool symmetricMemory) {
  // For non-symmetric algorithms, we use both input and output buffer pointers in the key.
  if (symmetricMemory) {
    size_t inputBytes, outputBytes;
    CUdeviceptr inputBasePtr, outputBasePtr;
    MSCCLPP_CUTHROW(cuMemGetAddressRange(&inputBasePtr, &inputBytes, (CUdeviceptr)inputBuffer));
    MSCCLPP_CUTHROW(cuMemGetAddressRange(&outputBasePtr, &outputBytes, (CUdeviceptr)outputBuffer));
    return AlgorithmCtxKey{(void*)inputBasePtr, (void*)outputBasePtr, inputBytes, outputBytes, 0};
  }
  return AlgorithmCtxKey{(void*)inputBuffer, outputBuffer, size, size, 0};
}

std::shared_ptr<void> AllreduceRsAgZeroCopy::initAllreduceContext(std::shared_ptr<Communicator> comm, const void* input,
                                                                  void* output, size_t size, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->ipcDomainNranks = getIpcDomainNranks(comm);

  ctx->memorySemaphores = this->semaphores_;

  // register input and output memories
  RegisteredMemory inputMemory = comm->registerMemory((void*)input, size, Transport::CudaIpc);
  RegisteredMemory outputMemory = comm->registerMemory(output, size, Transport::CudaIpc);
  this->inputMemories_.push_back(inputMemory);
  this->outputMemories_.push_back(outputMemory);

  auto remoteInputMemories = setupRemoteMemories(comm, ctx->rank, inputMemory);
  auto remoteOutputMemories = setupRemoteMemories(comm, ctx->rank, outputMemory);
  ctx->registeredMemories.insert(ctx->registeredMemories.end(), remoteInputMemories.begin(), remoteInputMemories.end());
  ctx->registeredMemories.insert(ctx->registeredMemories.end(), remoteOutputMemories.begin(),
                                 remoteOutputMemories.end());
  std::vector<void*> remoteMemoryHandles;
  for (const auto& remoteMemory : ctx->registeredMemories) {
    remoteMemoryHandles.push_back(remoteMemory.data());
  }
  ctx->remoteMemoryHandles = detail::gpuCallocShared<void*>(remoteMemoryHandles.size());
  gpuMemcpy(ctx->remoteMemoryHandles.get(), remoteMemoryHandles.data(), remoteMemoryHandles.size(),
            cudaMemcpyHostToDevice);

  // store local registered memories to ctx for lifetime management
  ctx->registeredMemories.push_back(inputMemory);
  ctx->registeredMemories.push_back(outputMemory);
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceRsAgZeroCopy::build() {
  auto self = std::make_shared<AllreduceRsAgZeroCopy>();
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_rsag_zero_copy", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras,
             DataType accumDtype) -> CommResult {
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
