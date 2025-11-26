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