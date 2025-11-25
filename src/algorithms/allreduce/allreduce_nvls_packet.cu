#include "algorithms/allreduce/allreduce_nvls_packet.hpp"
#include "algorithms/allreduce/common.hpp"
#include "algorithms/utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace algorithm {

__device__ uint32_t deviceFlag = 1;
template <Algorithm::Op OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceNvlsPacket([[maybe_unused]] const T* input, [[maybe_unused]] T* scratch, [[maybe_unused]] T* output,
                        [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
                        [[maybe_unused]] size_t nelems, [[maybe_unused]] size_t scratchBufferSize,
                        [[maybe_unused]] int rank, [[maybe_unused]] int worldSize, [[maybe_unused]] LL8Packet* flags) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t flag = deviceFlag;
  __syncthreads();
  if (threadIdx.x == 0) {
    flags[blockIdx.x].write(0, flag);
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
  if (blockIdx.x == 0 && threadIdx.x < gridDim.x) {
    flags[threadIdx.x].read(flag, -1);
  }
  if (blockIdx.x == 0) {
    __syncthreads();
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    deviceFlag++;
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

template <Op OpType, typename T>
struct AllreduceNvlsPacketAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void*, void*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t scratchBufferSize,
                          int rank, int, int worldSize, size_t inputSize, cudaStream_t stream, LL8Packet* flags,
                          uint32_t, int nBlocks, int nThreadsPerBlock) {
    allreduceNvlsPacket<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>((const T*)input, (T*)scratch, (T*)output,
                                                                             nvlsChannels, inputSize / sizeof(T),
                                                                             scratchBufferSize, rank, worldSize, flags);
    return cudaGetLastError();
  }
};

void AllreduceNvlsPacket::initialize(std::shared_ptr<mscclpp::Communicator>) {
  flags_ = mscclpp::detail::gpuCallocShared<LL8Packet>(16);
}

mscclpp::AlgorithmCtxKey AllreduceNvlsPacket::generateAllreduceContextKey(const void*, void*, size_t,
                                                                          mscclpp::DataType) {
  return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreduceNvlsPacket::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, const void*, void*, size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
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
                                                    cudaStream_t stream,
                                                    std::unordered_map<std::string, uintptr_t>& extra) {
  int op = *reinterpret_cast<int*>(extra.at("op"));
  std::pair<int, int> blockAndThreadNum = getBlockNumAndThreadNum(extra);
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize);
  }
  if (blockAndThreadNum.first > maxBlockNum_) {
    WARN("Block number %d exceeds the maximum limit %d", blockAndThreadNum.first, maxBlockNum_);
    return CommResult::commInvalidArgument;
  }
  AllreduceFunc allreduce = dispatch<AllreduceNvlsPacketAdapter>(static_cast<Algorithm::Op>(op), dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d", static_cast<int>(dtype));
    return CommResult::commInvalidArgument;
  }
  cudaError_t error = allreduce(
      input, this->scratchBuffer_, output, nullptr, nullptr, ctx->switchChannelDeviceHandles.get(),
      nullptr, 0, 0, this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize, inputSize, stream,
      this->flags_.get(), 0, blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsPacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::commUnhandledCudaError;
  }
  return CommResult::commSuccess;
}

std::shared_ptr<mscclpp::Algorithm> AllreduceNvlsPacket::build() {
  auto self = std::make_shared<AllreduceNvlsPacket>((uintptr_t)scratchBuffer_, scratchBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_nvls_packet", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype, cudaStream_t stream,
             std::unordered_map<std::string, uintptr_t>& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, stream, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->generateAllreduceContextKey(input, output, inputSize, dtype); });
}
}  // namespace algorithm
}  // namespace mscclpp