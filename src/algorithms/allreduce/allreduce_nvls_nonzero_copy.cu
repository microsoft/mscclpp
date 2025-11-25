// #include "algorithm_utils.hpp"
// #include "allreduce_common.hpp"
// #include "allreduce_nvls_nonzero_copy.hpp"
// #include "debug.h"

// namespace mscclpp {

// template <typename T>
// __global__ void __launch_bounds__(1024, 1)
//     allreduce10([[maybe_unused]] const void* src, [[maybe_unused]] void* scratch, [[maybe_unused]] void* dst,
//                 [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* memoryChannels,
//                 [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast, [[maybe_unused]] size_t size,
//                 [[maybe_unused]] size_t scratchBufferSize, [[maybe_unused]] int rank,
//                 [[maybe_unused]] int nRanksPerNode) {
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
//   constexpr int alignment = 16;
//   int nPeers = nRanksPerNode - 1;
//   int nBlocks = gridDim.x;
//   int nBlocksPerNvlsConn = nBlocks / NUM_NVLS_CONNECTION;
//   int bid = blockIdx.x;
//   size_t sizePerRank = size / nRanksPerNode;
//   size_t scratchSizePerRank = scratchBufferSize / nRanksPerNode;
//   const size_t maxSizePerBlock = ((sizePerRank + nBlocks - 1) / nBlocks + alignment - 1) / alignment * alignment;
//   size_t start = bid * maxSizePerBlock;
//   size_t end = min(start + maxSizePerBlock, sizePerRank);
//   size_t sizePerBlock = end - start;
//   auto* multicastPtr = multicast + bid / nBlocksPerNvlsConn;
//   size_t copyPerIter = 1024 * 16;
//   if (sizePerBlock >= 1024 * 64) {
//     copyPerIter = 1024 * 32;
//   }
//   size_t scratchSizePerBlock = (scratchSizePerRank / nBlocks) / copyPerIter * copyPerIter;
//   size_t blockScratchOffset = scratchSizePerBlock * bid + scratchSizePerRank * rank;
//   constexpr int NCOPY_WARPS = 14;
//   constexpr int NREDUCE_WARPS = 4;
//   constexpr int NRECV_COPY_WARPS = 14;
//   constexpr int endCopyWid = NCOPY_WARPS;
//   constexpr int startRecvCopyWid = NCOPY_WARPS;
//   constexpr int endRecvCopyWid = NCOPY_WARPS + NRECV_COPY_WARPS;
//   constexpr int endReduceWid = NCOPY_WARPS + NREDUCE_WARPS + NRECV_COPY_WARPS;
//   const int warpId = threadIdx.x / WARP_SIZE;
//   size_t nIter = sizePerBlock / copyPerIter;
//   size_t lastIterSize = copyPerIter;
//   if (sizePerBlock % copyPerIter != 0) {
//     nIter += 1;
//     lastIterSize = sizePerBlock % copyPerIter;
//   }

//   const size_t chanOffset = (nRanksPerNode - 1) * blockIdx.x * 2;
//   auto memoryChans = memoryChannels + chanOffset;
//   __shared__ mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel> channels[(MAX_NRANKS_PER_NODE - 1) * 2];
//   const int lid = threadIdx.x % WARP_SIZE;
//   if (lid < nPeers * 2) {
//     channels[lid] = memoryChans[lid];
//   }
//   __syncwarp();
//   for (int it = 0; it < nIter; it++) {
//     const size_t iterSize = (it == nIter - 1) ? lastIterSize : copyPerIter;
//     if (warpId < endCopyWid) {
//       int tidInCopy = threadIdx.x;
//       for (int i = 0; i < nRanksPerNode; i++) {
//         size_t offset = i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
//         size_t offsetScratch =
//             i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
//         char* srcData = (char*)src + offset;
//         char* dstData = (char*)scratch + offsetScratch;
//         mscclpp::copy(dstData, srcData, iterSize, tidInCopy, NCOPY_WARPS * WARP_SIZE);
//       }
//       asm volatile("bar.sync %0, %1;" ::"r"(0), "r"(NCOPY_WARPS * WARP_SIZE) : "memory");
//       if (tidInCopy < nPeers) {
//         channels[tidInCopy].signal();
//         channels[tidInCopy].wait();
//       }
//       asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
//     }
//     if (warpId >= endRecvCopyWid && warpId < endReduceWid) {
//       int tidInReduce = threadIdx.x - endRecvCopyWid * WARP_SIZE;
//       asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
//       T* mcBuff = (T*)multicastPtr->mcPtr;
//       size_t offset = blockScratchOffset + (it * copyPerIter) % scratchSizePerBlock;
//       handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, iterSize, tidInReduce, NREDUCE_WARPS * WARP_SIZE);
//       asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
//     }
//     if (warpId >= startRecvCopyWid && warpId < endRecvCopyWid) {
//       int tidInRecvCopy = threadIdx.x - startRecvCopyWid * WARP_SIZE;
//       asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
//       if (tidInRecvCopy < nPeers) {
//         channels[tidInRecvCopy + nPeers].signal();
//         channels[tidInRecvCopy + nPeers].wait();
//       }
//       asm volatile("bar.sync %0, %1;" ::"r"(3), "r"((NRECV_COPY_WARPS)*WARP_SIZE) : "memory");
//       for (int i = 0; i < nRanksPerNode; i++) {
//         size_t offset = i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
//         size_t offsetScratch =
//             i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
//         char* srcData = (char*)scratch + offsetScratch;
//         char* dstData = (char*)dst + offset;
//         mscclpp::copy(dstData, srcData, iterSize, tidInRecvCopy, NRECV_COPY_WARPS * WARP_SIZE);
//       }
//     }
//   }
// #endif
// }

// template <typename T>
// __global__ void __launch_bounds__(1024, 1)
//     allreduce11([[maybe_unused]] const void* src, [[maybe_unused]] void* scratch, [[maybe_unused]] void* dst,
//                 [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* memoryChannels,
//                 [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* switchChannels,
//                 [[maybe_unused]] size_t size, [[maybe_unused]] size_t scratchBufferSize, [[maybe_unused]] int rank,
//                 [[maybe_unused]] int nRanksPerNode) {
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
//   constexpr int alignment = 16;
//   int nPeers = nRanksPerNode - 1;
//   int nBlocksForCopy = nRanksPerNode * 2;
//   int nBlocksForReduce = nRanksPerNode;
//   int copyReduceRatio = nBlocksForCopy / nBlocksForReduce;
//   size_t scratchSizePerRank = scratchBufferSize / nRanksPerNode;
//   size_t sizePerRank = size / nRanksPerNode;
//   assert(sizePerRank % alignment == 0);
//   uint32_t sizePerBlock =
//       ((sizePerRank + (nBlocksForCopy - 1)) / nBlocksForCopy + alignment - 1) / alignment * alignment;
//   uint32_t lastBlockSize = sizePerRank - (nBlocksForCopy - 1) * sizePerBlock;
//   int bid = blockIdx.x;
//   int tid = threadIdx.x;
//   uint32_t unitSize = 1 << 17;
//   if (size <= 1024 * 1024 * 128) {
//     unitSize = 1 << 16;
//   }
//   int nIter = sizePerBlock / unitSize;
//   int nIterLastBlock = lastBlockSize / unitSize;
//   uint32_t lastIterSize = unitSize;
//   uint32_t lastBlockIterSize = unitSize;
//   if (sizePerBlock % unitSize != 0) {
//     nIter += 1;
//     lastIterSize = sizePerBlock % unitSize;
//   }
//   if (lastBlockSize % unitSize != 0) {
//     nIterLastBlock += 1;
//     lastBlockIterSize = lastBlockSize % unitSize;
//   }
//   if (bid == nBlocksForCopy - 1 || bid == 2 * nBlocksForCopy + nBlocksForReduce - 1) {
//     lastIterSize = lastBlockIterSize;
//     nIter = nIterLastBlock;
//   }
//   size_t scratchSizePerBlock = (scratchSizePerRank / nBlocksForCopy) / unitSize * unitSize;
//   size_t maxItersForScratch = scratchSizePerBlock / unitSize;
//   if (bid < nBlocksForCopy && tid == 0) {
//     deviceSemaphore[bid + 2 * nBlocksForCopy].set(maxItersForScratch);
//   }
//   for (int it = 0; it < nIter; it++) {
//     const uint32_t iterSize = (it == nIter - 1) ? lastIterSize : unitSize;
//     const uint32_t scratchIt = it % maxItersForScratch;
//     if (bid < nBlocksForCopy) {
//       if (tid == 0) {
//         deviceSemaphore[bid + 2 * nBlocksForCopy].acquire();
//       }
//       __syncthreads();
//       for (int i = 0; i < nRanksPerNode; i++) {
//         size_t blockOffset = it * unitSize + bid * sizePerBlock + i * sizePerRank;
//         uint32_t scratchOffset = scratchIt * unitSize + bid * scratchSizePerBlock + i * scratchSizePerRank;
//         char* srcData = (char*)src + blockOffset;
//         char* dstData = (char*)scratch + scratchOffset;
//         mscclpp::copy(dstData, srcData, iterSize, tid, blockDim.x);
//       }
//       __syncthreads();
//       if (tid < nPeers) {
//         int chanId = bid * nPeers + tid;
//         mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* channels = memoryChannels + chanId;
//         channels->signal();
//         channels->wait();
//       }
//       __syncthreads();
//       if (tid == 0) {
//         deviceSemaphore[bid].release();
//       }
//     }
//     if (bid >= nBlocksForCopy && bid < nBlocksForCopy + nBlocksForReduce) {
//       int bidForReduce = bid - nBlocksForCopy;
//       auto switchChannel = switchChannels + bidForReduce;
//       T* mcBuff = (T*)switchChannel->mcPtr;
//       for (int i = 0; i < copyReduceRatio; i++) {
//         int oriBid = bidForReduce * copyReduceRatio + i;
//         uint32_t offset = rank * scratchSizePerRank + scratchIt * unitSize + oriBid * scratchSizePerBlock;
//         uint32_t reduceIterSize = iterSize;
//         if ((oriBid == nBlocksForCopy - 1) && (it >= nIterLastBlock - 1)) {
//           if (it > nIterLastBlock - 1) {
//             continue;
//           }
//           reduceIterSize = lastBlockIterSize;
//         }
//         if (tid == 0) {
//           deviceSemaphore[oriBid].acquire();
//         }
//         __syncthreads();
//         handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, reduceIterSize, tid, blockDim.x);
//         __syncthreads();
//         if (tid == 0) {
//           deviceSemaphore[nBlocksForCopy + bidForReduce * copyReduceRatio + i].release();
//         }
//       }
//     }
//     if (bid >= nBlocksForCopy + nBlocksForReduce && bid < nBlocksForCopy + nBlocksForReduce + nBlocksForCopy) {
//       int bidForCopy = bid - nBlocksForCopy - nBlocksForReduce;
//       if (tid == 0) {
//         deviceSemaphore[bid - nBlocksForReduce].acquire();
//       }
//       __syncthreads();
//       if (tid < nPeers) {
//         int chanId = (bid - nBlocksForReduce) * nPeers + tid;
//         mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* channels = memoryChannels + chanId;
//         channels->signal();
//         channels->wait();
//       }
//       __syncthreads();
//       for (int i = 0; i < nRanksPerNode; i++) {
//         size_t blockOffset = it * unitSize + (bid - nBlocksForCopy - nBlocksForReduce) * sizePerBlock + i * sizePerRank;
//         uint32_t scratchOffset = scratchIt * unitSize +
//                                  (bid - nBlocksForCopy - nBlocksForReduce) * scratchSizePerBlock +
//                                  i * scratchSizePerRank;
//         char* srcData = (char*)scratch + scratchOffset;
//         char* dstData = (char*)dst + blockOffset;
//         mscclpp::copy(dstData, srcData, iterSize, tid, blockDim.x);
//       }
//       __syncthreads();
//       if (tid == 0) {
//         deviceSemaphore[bidForCopy + 2 * nBlocksForCopy].release();
//       }
//     }
//   }
//   if (bid < nBlocksForCopy && tid == 0) {
//     deviceSemaphore[bid + 2 * nBlocksForCopy].set(0);
//   }
// #endif
// }

// template <Op OpType, typename T>
// struct NvlsWithCopyAdapter {
//   static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void*,
//                           mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
//                           mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t scratchBufferSize,
//                           int rank, int nRanksPerNode, int, size_t inputSize, cudaStream_t stream, uint32_t*, uint32_t*,
//                           uint32_t*, uint32_t) {
// #if defined(__CUDA_ARCH__)  // Skip the __CUDA_ARCH__ < 1000 since FP8 has not been supported for NVLS
//     if constexpr (std::is_same_v<T, __fp8_e4m3> || std::is_same_v<T, __fp8_e5m2>) {
//       return cudaErrorNotSupported;
//     } else
// #endif
//     {
//       using ChannelType = mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>;
//       if (inputSize < (1 << 24)) {
//         int nBlocks = nRanksPerNode * 4;
//         int nThreadsPerBlock = 1024;
//         allreduce10<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(input, scratch, output, (ChannelType*)memoryChannels,
//                                                                  nvlsChannels, inputSize, scratchBufferSize, rank,
//                                                                  nRanksPerNode);
//       } else {
//         int nBlocks = nRanksPerNode * 5;
//         int nThreadsPerBlock = 1024;
//         allreduce11<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(input, scratch, output, (ChannelType*)memoryChannels,
//                                                                  nvlsChannels, inputSize, scratchBufferSize, rank,
//                                                                  nRanksPerNode);
//       }
//       return cudaGetLastError();
//     }
//   }
// };

// void AllreduceNvlsWithCopy::initialize(std::shared_ptr<mscclpp::Communicator> comm) {
//   nSwitchChannels_ = 8;
//   int nBaseChannels = 64;
//   this->conns_ = setupConnections(comm);
//   // setup semaphores
//   std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores =
//       setupMemorySemaphores(comm, this->conns_, nBaseChannels);
//   // setup base memory channels
//   this->baseChannels_ = setupBaseMemoryChannels(this->conns_, memorySemaphores, nBaseChannels);
//   this->memoryChannelsDeviceHandle_ = setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
// }

// CommResult AllreduceNvlsWithCopy::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx,
//                                                       const void* input, void* output, size_t inputSize,
//                                                       mscclpp::DataType dtype, cudaStream_t stream,
//                                                       std::unordered_map<std::string, uintptr_t>&) {
//   AllreduceFunc allreduce = dispatch<NvlsWithCopyAdapter>(Algorithm::Op::SUM, dtype);
//   if (!allreduce) {
//     WARN("Unsupported operation or data type for allreduce, dtype=%d", static_cast<int>(dtype));
//     return CommResult::commInvalidArgument;
//   }
//   cudaError_t error =
//       allreduce(input, this->scratchBuffer_.lock().get(), output, this->memoryChannelsDeviceHandle_.get(), nullptr,
//                 ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0, this->scratchBufferSize_, ctx->rank,
//                 ctx->nRanksPerNode, ctx->workSize, inputSize, stream, nullptr, nullptr, nullptr, 0);
//   if (error != cudaSuccess) {
//     WARN("AllreduceNvlsWithCopy failed with error: %s", cudaGetErrorString(error));
//     return CommResult::commUnhandledCudaError;
//   }
//   return CommResult::commSuccess;
// }

// mscclpp::AlgorithmCtxKey AllreduceNvlsWithCopy::generateAllreduceContextKey(const void*, void*, size_t,
//                                                                             mscclpp::DataType) {
//   return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
// }

// std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceNvlsWithCopy::initAllreduceContext(
//     std::shared_ptr<mscclpp::Communicator> comm, const void*, void*, size_t, mscclpp::DataType) {
//   auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
//   ctx->rank = comm->bootstrap()->getRank();
//   ctx->workSize = comm->bootstrap()->getNranks();
//   ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

//   // setup channels
//   ctx->nvlsConnections = setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
//   ctx->switchChannels =
//       setupNvlsChannels(ctx->nvlsConnections, this->scratchBuffer_.lock().get(), scratchBufferSize_, nSwitchChannels_);
//   ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
//   return ctx;
// }

// std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsWithCopy::build() {
//   auto self = std::make_shared<AllreduceNvlsWithCopy>(scratchBuffer_.lock(), scratchBufferSize_);
//   return std::make_shared<mscclpp::NativeAlgorithm>(
//       "default_allreduce_nvls_with_copy", "allreduce",
//       [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
//       [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
//              [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype, cudaStream_t stream,
//              std::unordered_map<std::string, uintptr_t>& extras) {
//         return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, stream, extras);
//       },
//       [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
//              [[maybe_unused]] size_t outputSize,
//              mscclpp::DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
//       [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
//              mscclpp::DataType dtype) { return self->generateAllreduceContextKey(input, output, inputSize, dtype); });
// }

// }  // namespace mscclpp