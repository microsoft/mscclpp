// #include "algorithm_utils.hpp"
// #include "allreduce_common.hpp"
// #include "allreduce_nvls_packet.hpp"
// #include "debug.h"

// namespace mscclpp {

// inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize) {
//   int blockNum = 8;
//   int threadNum = 1024;
//   if (inputSize <= (1 << 13)) {
//     blockNum = 4;
//     threadNum = 512;
//   }
//   return {blockNum, threadNum};
// }

// template <Op OpType, typename T>
// struct AllreduceNvlsPacketAdapter {
//   static cudaError_t call(const void* input, void* scratch, void* output, void*, void*,
//                           mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
//                           mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t scratchBufferSize,
//                           int rank, int, int worldSize, size_t inputSize, cudaStream_t stream, uint32_t* deviceFlag,
//                           uint32_t*, uint32_t*, uint32_t, int nBlocks, int nThreadsPerBlock) {
//     allreduceNvlsPacket<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
//         (const T*)input, (T*)scratch, (T*)output, nvlsChannels, inputSize / sizeof(T), scratchBufferSize, rank,
//         worldSize, deviceFlag);
//     return cudaGetLastError();
//   }
// };

// void AllreduceNvlsPacket::initialize(std::shared_ptr<mscclpp::Communicator>) {
//   deviceFlag_ = mscclpp::detail::gpuCallocShared<uint32_t>(16);
//   std::vector<uint32_t> initFlag(16);
//   for (int i = 0; i < 16; ++i) {
//     initFlag[i] = 1;
//   }
//   mscclpp::gpuMemcpy<uint32_t>(deviceFlag_.get(), initFlag.data(), 16, cudaMemcpyHostToDevice);
// }

// mscclpp::AlgorithmCtxKey AllreduceNvlsPacket::generateAllreduceContextKey(const void*, void*, size_t,
//                                                                           mscclpp::DataType) {
//   return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
// }

// std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceNvlsPacket::initAllreduceContext(
//     std::shared_ptr<mscclpp::Communicator> comm, const void*, void*, size_t, mscclpp::DataType) {
//   auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
//   ctx->rank = comm->bootstrap()->getRank();
//   ctx->workSize = comm->bootstrap()->getNranks();
//   ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

//   // setup channels
//   int nSwitchChannels = 1;
//   ctx->nvlsConnections = setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels);
//   ctx->switchChannels = setupNvlsChannels(ctx->nvlsConnections, this->scratchBuffer_.lock().get(),
//                                           this->scratchBufferSize_, nSwitchChannels);
//   ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
//   return ctx;
// }

// CommResult AllreduceNvlsPacket::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
//                                                     void* output, size_t inputSize, mscclpp::DataType dtype,
//                                                     cudaStream_t stream,
//                                                     std::unordered_map<std::string, uintptr_t>& extra) {
//   int op = *reinterpret_cast<int*>(extra.at("op"));
//   std::pair<int, int> blockAndThreadNum = getBlockNumAndThreadNum(extra);
//   if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
//     blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize);
//   }
//   if (blockAndThreadNum.first > maxBlockNum_) {
//     WARN("Block number %d exceeds the maximum limit %d", blockAndThreadNum.first, maxBlockNum_);
//     return CommResult::commInvalidArgument;
//   }
//   AllreduceFunc allreduce = dispatch<AllreduceNvlsPacketAdapter>(static_cast<Algorithm::Op>(op), dtype);
//   if (!allreduce) {
//     WARN("Unsupported operation or data type for allreduce, dtype=%d", static_cast<int>(dtype));
//     return CommResult::commInvalidArgument;
//   }
//   cudaError_t error = allreduce(
//       input, this->scratchBuffer_.lock().get(), output, nullptr, nullptr, ctx->switchChannelDeviceHandles.get(),
//       nullptr, 0, 0, this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize, inputSize, stream,
//       this->deviceFlag_.get(), nullptr, nullptr, 0, blockAndThreadNum.first, blockAndThreadNum.second);
//   if (error != cudaSuccess) {
//     WARN("AllreduceNvlsPacket failed with error: %s", cudaGetErrorString(error));
//     return CommResult::commUnhandledCudaError;
//   }
//   return CommResult::commSuccess;
// }

// std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsPacket::build() {
//   auto self = std::make_shared<AllreduceNvlsPacket>(scratchBuffer_.lock(), scratchBufferSize_);
//   return std::make_shared<mscclpp::NativeAlgorithm>(
//       "default_allreduce_nvls_packet", "allreduce",
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