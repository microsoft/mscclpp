// // Copyright (c) Microsoft Corporation.
// // Licensed under the MIT license.

// #ifndef MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_
// #define MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_

// #include <mscclpp/algorithm.hpp>

// namespace mscclpp {
// template <Algorithm::Op OpType, typename T>
// __global__ void __launch_bounds__(1024, 1)
//     allreduceNvlsPacket([[maybe_unused]] const T* input, [[maybe_unused]] T* scratch, [[maybe_unused]] T* output,
//                         [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
//                         [[maybe_unused]] size_t nelems, [[maybe_unused]] size_t scratchBufferSize,
//                         [[maybe_unused]] int rank, [[maybe_unused]] int worldSize,
//                         [[maybe_unused]] uint32_t* deviceFlag) {
//   #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
//   uint32_t flag = deviceFlag[blockIdx.x];
//   size_t scratchBaseOffset = (flag % 2) ? scratchBufferSize / 2 : 0;
//   uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//   uint32_t nPktPerRank = nelems / worldSize / (sizeof(mscclpp::LL8Packet::Payload) / sizeof(T));
//   mscclpp::LL8Packet* multiPkt =
//       (mscclpp::LL8Packet*)((char*)multicast->mcPtr + scratchBaseOffset) + rank * worldSize * nPktPerRank;
//   uint* src = (uint*)(input);
//   uint* dst = (uint*)(output);
//   mscclpp::LL8Packet* scratchPkt = (mscclpp::LL8Packet*)((char*)scratch + scratchBaseOffset);
//   for (uint32_t i = tid; i < nPktPerRank * worldSize; i += blockDim.x * gridDim.x) {
//     mscclpp::LL8Packet pkt(src[i], flag);
//     mscclpp::SwitchChannelDeviceHandle::multimemStore(*(mscclpp::f32x2*)(&pkt), multiPkt + i);
//   }
//   for (uint32_t i = tid; i < nPktPerRank * worldSize; i += blockDim.x * gridDim.x) {
//     uint data = src[i];
//     for (int peer = 0; peer < worldSize; peer++) {
//       if (peer == rank) {
//         continue;
//       }
//       uint val = scratchPkt[peer * worldSize * nPktPerRank + i].read(flag);
//       data = cal_vectors<T, OpType>(data, val);
//     }
//     dst[i] = data;
//   }
//   if (threadIdx.x == 0) {
//     deviceFlag[blockIdx.x] = deviceFlag[blockIdx.x] + 1;
//   }
//   #endif
// }

// class AllreduceNvlsPacket : public mscclpp::AlgorithmBuilder {
//  public:
//   AllreduceNvlsPacket(std::shared_ptr<void> scratchBuffer, size_t scratchBufferSize)
//       : scratchBuffer_(std::static_pointer_cast<char>(scratchBuffer)), scratchBufferSize_(scratchBufferSize){};
//   std::shared_ptr<mscclpp::Algorithm> build() override;

//  private:
//   void initialize(std::shared_ptr<mscclpp::Communicator> comm);
//   CommResult allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output,
//                                  size_t inputSize, mscclpp::DataType dtype, cudaStream_t stream,
//                                  std::unordered_map<std::string, uintptr_t>& extras);

//   std::shared_ptr<mscclpp::AlgorithmCtx> initAllreduceContext(std::shared_ptr<mscclpp::Communicator> comm, const void*,
//                                                               void* output, size_t, mscclpp::DataType);
//   mscclpp::AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, mscclpp::DataType);

//   size_t scratchBufferSize_;
//   std::weak_ptr<char> scratchBuffer_;
//   const size_t nvlsBufferSize_ = (1 << 30);
//   const int maxBlockNum_ = 16;
//   std::shared_ptr<uint32_t> deviceFlag_;
// };
// }  // namespace mscclpp
// #endif