// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "alltoallv/alltoallv_fullmesh.hpp"
#include "alltoallv/alltoallv_kernel.hpp"
#include "collective_utils.hpp"

#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>

#include <algorithm>

namespace mscclpp {
namespace collective {

#if defined(__HIP_PLATFORM_AMD__)
#define ALLTOALLV_WARP_SIZE 64
#else
#define ALLTOALLV_WARP_SIZE 32
#endif

using MultiNodeMode = AlltoallvFullmesh::MultiNodeMode;

// Context to hold all necessary state for alltoallv execution
struct AllToAllVContext {
  int rank;
  int worldSize;
  int nRanksPerNode;

  // MemoryChannel (CudaIpc) — used for intra-node (always) and cross-node (NVSwitch mode)
  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;

  // PortChannel (IB) — used for cross-node peers in IB mode only
  std::shared_ptr<ProxyService> proxyService;
  std::vector<PortChannel> portChannels;
  std::shared_ptr<PortChannelDeviceHandle> portChannelDeviceHandles;

  // Peer locality map (IB mode only)
  std::shared_ptr<int> d_peerIsLocal;           // GPU array [nPeers]
  std::shared_ptr<int> d_peerToPortChannelIdx;  // GPU array [nPeers]

  // Staging buffers (NVSwitch mode only): allocated via GpuBuffer (cuMemCreate → Fabric handles)
  bool useStaging;
  std::shared_ptr<GpuBuffer<char>> inputStaging;
  std::shared_ptr<GpuBuffer<char>> outputStaging;

  // Which kernel dispatch path to use
  AlltoallvFullmesh::MultiNodeMode mode;

  std::shared_ptr<DeviceSyncer> deviceSyncer;
};

AlltoallvFullmesh::~AlltoallvFullmesh() = default;

std::shared_ptr<Algorithm> AlltoallvFullmesh::build() {
  // Create a new shared_ptr that owns the object to keep it alive
  // This ensures the lambdas capturing 'self' have a valid object
  auto self = std::make_shared<AlltoallvFullmesh>();

  std::shared_ptr<Algorithm> alltoallvAlgo = std::make_shared<NativeAlgorithm>(
      "default_alltoallv_fullmesh", "alltoallv",  // name, collective (was swapped before)
      // Initialize function
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      // Kernel execution function
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             size_t outputSize, DataType dtype, [[maybe_unused]] ReduceOp op, cudaStream_t stream,
             int nBlocks, int nThreadsPerBlock,
             const std::unordered_map<std::string, uintptr_t>& extras) {
        return self->alltoallvKernelFunc(ctx, input, output, inputSize, outputSize, dtype, stream,
                                         nBlocks, nThreadsPerBlock, extras);
      },
      // Context initialization function
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             size_t outputSize, DataType dtype) {
        return self->initAlltoallvContext(comm, input, output, inputSize, outputSize, dtype);
      },
      // Context key generation function
      [self](const void* input, void* output, size_t inputSize, size_t outputSize, DataType dtype) {
        return self->generateAlltoallvContextKey(input, output, inputSize, outputSize, dtype);
      });

  return alltoallvAlgo;
}

void AlltoallvFullmesh::initialize(std::shared_ptr<Communicator> comm) {
  worldSize_ = comm->bootstrap()->getNranks();
  int rank = comm->bootstrap()->getRank();
  int nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  int localGpuIdx = rank % nRanksPerNode;
  bool isMultiNode = (worldSize_ > nRanksPerNode);

  if (!isMultiNode) {
    // ── Single-node: CudaIpc for all peers ─────────────────────────────
    multiNodeMode_ = MultiNodeMode::SingleNode;
    this->conns_ = setupConnections(comm);
  } else if (isNvlsSupported()) {
    // ── GB200 NVSwitch: CudaIpc for ALL peers + staging GpuBuffers ─────
    // GpuBuffer uses cuMemCreate → Fabric handles → cross-node CudaIpc works.
    multiNodeMode_ = MultiNodeMode::NVSwitch;
    this->conns_ = setupConnections(comm);
  } else {
    // ── IB: CudaIpc intra-node + IB inter-node ────────────────────────
    // For non-NVSwitch systems (H100 etc.) where CudaIpc doesn't work cross-node.
    if (getIBDeviceCount() <= 0) {
      throw Error("Multi-node alltoallv requires IB transport but no IB devices found. "
                  "Ensure IB drivers are loaded and devices are available.",
                  ErrorCode::InvalidUsage);
    }
    multiNodeMode_ = MultiNodeMode::IB;
    this->conns_ = setupHybridConnections(comm, localGpuIdx);
  }
}

CommResult AlltoallvFullmesh::alltoallvKernelFunc(
    const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
    [[maybe_unused]] size_t outputSize, [[maybe_unused]] DataType dtype, cudaStream_t stream,
    [[maybe_unused]] int nBlocks, int nThreadsPerBlock,
    const std::unordered_map<std::string, uintptr_t>& extras) {

  auto algoCtx = std::static_pointer_cast<AllToAllVContext>(ctx);
  int rank = algoCtx->rank;
  int worldSize = algoCtx->worldSize;

  // Extract send/recv counts and displacements from extras
  auto it_sendCounts = extras.find("sendCounts");
  auto it_sendDispls = extras.find("sendDispls");
  auto it_recvCounts = extras.find("recvCounts");
  auto it_recvDispls = extras.find("recvDispls");
  auto it_remoteRecvDispls = extras.find("remoteRecvDispls");

  if (it_sendCounts == extras.end() || it_sendDispls == extras.end() ||
      it_recvCounts == extras.end() || it_recvDispls == extras.end() ||
      it_remoteRecvDispls == extras.end()) {
    return CommResult::CommInternalError;
  }

  const size_t* d_sendCounts = reinterpret_cast<const size_t*>(it_sendCounts->second);
  const size_t* d_sendDispls = reinterpret_cast<const size_t*>(it_sendDispls->second);
  const size_t* d_recvCounts = reinterpret_cast<const size_t*>(it_recvCounts->second);
  const size_t* d_recvDispls = reinterpret_cast<const size_t*>(it_recvDispls->second);
  const size_t* d_remoteRecvDispls = reinterpret_cast<const size_t*>(it_remoteRecvDispls->second);

  // Use maximum threads (1024) for best bandwidth utilization
  const int threadsPerBlock = (nThreadsPerBlock > 0 && nThreadsPerBlock <= 1024) ? nThreadsPerBlock : 1024;

  int nPeers = worldSize - 1;
  if (nPeers < 1) nPeers = 1;

  // Determine send/recv buffer pointers.
  // NVSwitch mode: copy PyTorch data to/from GpuBuffer staging buffers.
  const void* sendBuff = input;
  void* recvBuff = output;

  if (algoCtx->useStaging) {
    sendBuff = algoCtx->inputStaging->data();
    recvBuff = algoCtx->outputStaging->data();
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        const_cast<void*>(sendBuff), input,
        inputSize, cudaMemcpyDeviceToDevice, stream));
  }

  if (algoCtx->mode == MultiNodeMode::IB) {
    // ── IB mode: PortChannel kernel for ALL peers ──────────────────────
    // PortChannel handles both CudaIpc (intra) and IB (inter) connections
    // via the ProxyService proxy thread.
    int numBlocks = nPeers;
    alltoallvPortChannelKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        algoCtx->portChannelDeviceHandles.get(),
        rank, worldSize,
        sendBuff, recvBuff,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  } else {
    // ── SingleNode / NVSwitch mode: MemoryChannel kernel ───────────────
    constexpr size_t SIZE_THRESHOLD = 1 << 20;  // 1MB
    size_t avgMsgSize = inputSize / worldSize;

    if (avgMsgSize < SIZE_THRESHOLD) {
      int numBlocks = nPeers;
      alltoallvPeerParallelKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
          algoCtx->memoryChannelDeviceHandles.get(),
          algoCtx->deviceSyncer.get(),
          rank, worldSize,
          sendBuff, recvBuff,
          d_sendCounts, d_sendDispls,
          d_recvCounts, d_recvDispls,
          d_remoteRecvDispls);
    } else {
      int blocksPerPeer = (nBlocks > 0 && nBlocks <= 128)
          ? ((nBlocks + nPeers - 1) / nPeers)
          : ALLTOALLV_DEFAULT_BLOCKS_PER_PEER;
      int numBlocks = nPeers * blocksPerPeer;
      if (numBlocks > 128) numBlocks = (128 / nPeers) * nPeers;
      if (numBlocks < nPeers) numBlocks = nPeers;
      alltoallvPeerParallelKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
          algoCtx->memoryChannelDeviceHandles.get(),
          algoCtx->deviceSyncer.get(),
          rank, worldSize,
          sendBuff, recvBuff,
          d_sendCounts, d_sendDispls,
          d_recvCounts, d_recvDispls,
          d_remoteRecvDispls);
    }
  }

  if (algoCtx->useStaging) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(
        output, recvBuff,
        outputSize, cudaMemcpyDeviceToDevice, stream));
  }

  if (cudaGetLastError() == cudaSuccess) {
    return CommResult::CommSuccess;
  }
  return CommResult::CommInternalError;
}

std::shared_ptr<void> AlltoallvFullmesh::initAlltoallvContext(
    std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
    size_t outputSize, [[maybe_unused]] DataType dtype) {

  auto ctx = std::make_shared<AllToAllVContext>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->worldSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  ctx->mode = this->multiNodeMode_;
  ctx->useStaging = (ctx->mode == MultiNodeMode::NVSwitch);

  int rank = ctx->rank;
  int localGpuIdx = rank % ctx->nRanksPerNode;

  if (ctx->mode == MultiNodeMode::NVSwitch) {
    // ── NVSwitch (GB200): staging GpuBuffers + CudaIpc MemoryChannel for all peers
    ctx->inputStaging = std::make_shared<GpuBuffer<char>>(inputSize);
    ctx->outputStaging = std::make_shared<GpuBuffer<char>>(outputSize);

    TransportFlags allTransports = Transport::CudaIpc;
    RegisteredMemory inputBufRegMem = comm->registerMemory(
        ctx->inputStaging->data(), ctx->inputStaging->bytes(), allTransports);
    RegisteredMemory outputBufRegMem = comm->registerMemory(
        ctx->outputStaging->data(), ctx->outputStaging->bytes(), allTransports);

    std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, rank, outputBufRegMem);

    constexpr int nChannelsPerConnection = 1;
    ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    ctx->memoryChannels = setupMemoryChannels(
        this->conns_, ctx->memorySemaphores, remoteOutputMemories, inputBufRegMem, nChannelsPerConnection);
    ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

    ctx->registeredMemories = std::move(remoteOutputMemories);
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

  } else if (ctx->mode == MultiNodeMode::IB) {
    // ── IB: PortChannel for ALL peers (CudaIpc intra + IB inter connections)
    TransportFlags allTransports = Transport::CudaIpc | getIBTransportForGpu(localGpuIdx);
    RegisteredMemory inputBufRegMem = comm->registerMemory((void*)input, inputSize, allTransports);
    RegisteredMemory outputBufRegMem = comm->registerMemory(output, outputSize, allTransports);

    std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, rank, outputBufRegMem);

    ctx->proxyService = std::make_shared<ProxyService>();
    ctx->portChannels = setupAllPortChannels(
        ctx->proxyService, *comm, this->conns_, remoteOutputMemories, inputBufRegMem);
    ctx->portChannelDeviceHandles = setupPortChannelDeviceHandles(ctx->portChannels);
    ctx->proxyService->startProxy(true);

    ctx->registeredMemories = std::move(remoteOutputMemories);
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

  } else {
    // ── SingleNode: CudaIpc MemoryChannel (direct PyTorch buffers)
    TransportFlags allTransports = Transport::CudaIpc;
    RegisteredMemory inputBufRegMem = comm->registerMemory((void*)input, inputSize, allTransports);
    RegisteredMemory outputBufRegMem = comm->registerMemory(output, outputSize, allTransports);

    std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, rank, outputBufRegMem);

    constexpr int nChannelsPerConnection = 1;
    ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    ctx->memoryChannels = setupMemoryChannels(
        this->conns_, ctx->memorySemaphores, remoteOutputMemories, inputBufRegMem, nChannelsPerConnection);
    ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

    ctx->registeredMemories = std::move(remoteOutputMemories);
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);
  }

  // Allocate GPU DeviceSyncer for multi-block grid-wide barrier
  ctx->deviceSyncer = mscclpp::detail::gpuCallocShared<DeviceSyncer>();

  return ctx;
}

AlgorithmCtxKey AlltoallvFullmesh::generateAlltoallvContextKey(
    const void* input, void* output, size_t inputSize, size_t outputSize,
    [[maybe_unused]] DataType dtype) {
  return {(void*)input, output, inputSize, outputSize, 0};
}

#undef ALLTOALLV_WARP_SIZE

}  // namespace collective
}  // namespace mscclpp
