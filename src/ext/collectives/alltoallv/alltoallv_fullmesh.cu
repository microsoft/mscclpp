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

#include "debug.h"

namespace mscclpp {
namespace collective {

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

  self->algo_ = alltoallvAlgo;
  return alltoallvAlgo;
}

void AlltoallvFullmesh::initialize(std::shared_ptr<Communicator> comm) {
  worldSize_ = comm->bootstrap()->getNranks();
  int rank = comm->bootstrap()->getRank();
  int nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  int localGpuIdx = rank % nRanksPerNode;
  bool isMultiNode = (worldSize_ > nRanksPerNode);
  bool nvlsSupported = isNvlsSupported();
  int ibDevCount = getIBDeviceCount();

  // Detect compute capability to distinguish NVSwitch topologies:
  //   SM 10.x (Blackwell/GB200): NVSwitch fabric can span across nodes (MNNVLS),
  //     so CudaIpc works cross-node → prefer NVSwitch mode.
  //   SM 9.x  (Hopper/H100):     NVSwitch is intra-node only,
  //     CudaIpc cannot map cross-node memory → must use IB for cross-node.
  int computeCapabilityMajor = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&computeCapabilityMajor,
                                            cudaDevAttrComputeCapabilityMajor, localGpuIdx));

  INFO(MSCCLPP_COLL, "[alltoallv][rank %d] initialize: worldSize=%d, nRanksPerNode=%d, "
       "isMultiNode=%d, isNvlsSupported=%d, ibDevCount=%d, localGpuIdx=%d, computeCapabilityMajor=%d",
       rank, worldSize_, nRanksPerNode, isMultiNode, nvlsSupported, ibDevCount, localGpuIdx,
       computeCapabilityMajor);

  if (!isMultiNode) {
    multiNodeMode_ = MultiNodeMode::SingleNode;
    this->conns_ = setupConnections(comm);
  } else if (nvlsSupported && computeCapabilityMajor >= 10) {
    // Blackwell/GB200 (SM 10.x+): NVSwitch fabric spans across nodes (MNNVLS).
    // CudaIpc works cross-node → use NVSwitch mode for all peers.
    multiNodeMode_ = MultiNodeMode::NVSwitch;
    this->conns_ = setupConnections(comm);
  } else if (ibDevCount > 0) {
    // Hopper/Ampere (SM 9.x/8.x) or no NVLS: NVSwitch is intra-node only.
    // Use IB (PortChannel) for cross-node, CudaIpc for intra-node.
    multiNodeMode_ = MultiNodeMode::IB;
    this->conns_ = setupHybridConnections(comm, localGpuIdx);
  } else {
    throw Error("Multi-node alltoallv requires either IB transport or cross-node NVSwitch (GB200+). "
                "On Hopper/Ampere, ensure IB drivers are loaded. On Blackwell, ensure NVSwitch is "
                "properly configured.",
                ErrorCode::InvalidUsage);
  }

  const char* modeStr = (multiNodeMode_ == MultiNodeMode::SingleNode) ? "SingleNode" :
                        (multiNodeMode_ == MultiNodeMode::NVSwitch) ? "NVSwitch" : "IB";
  INFO(MSCCLPP_COLL, "[alltoallv][rank %d] mode=%s, connections=%zu",
       rank, modeStr, this->conns_.size());
  for (size_t i = 0; i < this->conns_.size(); ++i) {
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d]   conn[%zu] transport=%d",
         rank, i, (int)this->conns_[i].transport());
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
    // ── IB mode: Hybrid kernel ─────────────────────────────────────────
    // MemoryChannel (direct NVLink) for intra-node peers,
    // PortChannel (CPU proxy → RDMA) for inter-node peers.
    int numBlocks = nPeers;
    alltoallvHybridKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        algoCtx->memoryChannelDeviceHandles.get(),
        algoCtx->portChannelDeviceHandles.get(),
        algoCtx->d_peerIsLocal.get(),
        algoCtx->d_peerToPortChannelIdx.get(),
        algoCtx->deviceSyncer.get(),
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

  // NOTE: Do NOT reset() here.  The periodic reset was destroying the
  // cached context (MemoryChannels, semaphores) while inter-GPU signaling
  // was still in progress, causing semaphore epoch mismatch and eventually
  // illegal memory access.  With persistent fixed-size buffers the context
  // key is stable, so the cached context is valid for the lifetime of the
  // communicator.

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
  const char* modeStr = (ctx->mode == MultiNodeMode::SingleNode) ? "SingleNode" :
                        (ctx->mode == MultiNodeMode::NVSwitch) ? "NVSwitch" : "IB";
  INFO(MSCCLPP_COLL, "[alltoallv][rank %d] initContext: mode=%s, useStaging=%d, "
       "input=%p (%zu B), output=%p (%zu B), localGpuIdx=%d",
       rank, modeStr, ctx->useStaging, input, inputSize, output, outputSize, localGpuIdx);

  if (ctx->mode == MultiNodeMode::NVSwitch) {
    // ── NVSwitch (GB200): staging GpuBuffers + CudaIpc MemoryChannel for all peers
    ctx->inputStaging = std::make_shared<GpuBuffer<char>>(inputSize);
    ctx->outputStaging = std::make_shared<GpuBuffer<char>>(outputSize);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] NVSwitch staging: input=%p (%zu B), output=%p (%zu B)",
         rank, ctx->inputStaging->data(), inputSize, ctx->outputStaging->data(), outputSize);

    TransportFlags allTransports = Transport::CudaIpc;
    RegisteredMemory inputBufRegMem = comm->registerMemory(
        ctx->inputStaging->data(), ctx->inputStaging->bytes(), allTransports);
    RegisteredMemory outputBufRegMem = comm->registerMemory(
        ctx->outputStaging->data(), ctx->outputStaging->bytes(), allTransports);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] NVSwitch: registered input=%p, output=%p",
         rank, inputBufRegMem.data(), outputBufRegMem.data());

    std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, rank, outputBufRegMem);
    for (size_t i = 0; i < remoteOutputMemories.size(); ++i) {
      INFO(MSCCLPP_COLL, "[alltoallv][rank %d] NVSwitch: remoteOutput[%zu] data=%p, size=%zu",
           rank, i, remoteOutputMemories[i].data(), remoteOutputMemories[i].size());
      if (remoteOutputMemories[i].data() == nullptr) {
        INFO(MSCCLPP_COLL, "[alltoallv][rank %d] ERROR: remoteOutput[%zu] has NULL data pointer! "
             "Cross-node CudaIpc mapping failed.", rank, i);
      }
    }

    constexpr int nChannelsPerConnection = 1;
    ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] NVSwitch: %zu semaphores created",
         rank, ctx->memorySemaphores.size());
    ctx->memoryChannels = setupMemoryChannels(
        this->conns_, ctx->memorySemaphores, remoteOutputMemories, inputBufRegMem, nChannelsPerConnection);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] NVSwitch: %zu memoryChannels created",
         rank, ctx->memoryChannels.size());
    ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

    ctx->registeredMemories = std::move(remoteOutputMemories);
    ctx->registeredMemories.push_back(inputBufRegMem);
    ctx->registeredMemories.push_back(outputBufRegMem);

  } else if (ctx->mode == MultiNodeMode::IB) {
    // ── IB hybrid: MemoryChannel (intra-node) + PortChannel (inter-node) ──
    TransportFlags allTransports = Transport::CudaIpc | getIBTransportForGpu(localGpuIdx);
    RegisteredMemory inputBufRegMem = comm->registerMemory((void*)input, inputSize, allTransports);
    RegisteredMemory outputBufRegMem = comm->registerMemory(output, outputSize, allTransports);

    std::vector<RegisteredMemory> remoteOutputMemories = setupRemoteMemories(comm, rank, outputBufRegMem);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] IB hybrid: input=%p (%zu B), output=%p (%zu B), remotes=%zu",
         rank, input, inputSize, output, outputSize, remoteOutputMemories.size());

    // Build peer locality map and per-type channel arrays
    int nPeers = ctx->worldSize - 1;
    int thisNode = rank / ctx->nRanksPerNode;
    std::vector<int> peerIsLocal(nPeers, 0);
    std::vector<int> peerToPortChIdx(nPeers, -1);
    int portChCount = 0;
    for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
      int peer = peerIdx < rank ? peerIdx : peerIdx + 1;
      if (peer / ctx->nRanksPerNode == thisNode) {
        peerIsLocal[peerIdx] = 1;
      } else {
        peerToPortChIdx[peerIdx] = portChCount++;
      }
    }
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] IB hybrid: nPeers=%d, localPeers=%d, remotePeers=%d",
         rank, nPeers, nPeers - portChCount, portChCount);

    // Copy locality arrays to GPU
    ctx->d_peerIsLocal = mscclpp::detail::gpuCallocShared<int>(nPeers);
    ctx->d_peerToPortChannelIdx = mscclpp::detail::gpuCallocShared<int>(nPeers);
    mscclpp::gpuMemcpy<int>(ctx->d_peerIsLocal.get(), peerIsLocal.data(), nPeers, cudaMemcpyHostToDevice);
    mscclpp::gpuMemcpy<int>(ctx->d_peerToPortChannelIdx.get(), peerToPortChIdx.data(), nPeers, cudaMemcpyHostToDevice);

    // MemoryChannel for intra-node CudaIpc connections (direct NVLink put)
    constexpr int nChannelsPerConnection = 1;
    ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    ctx->memoryChannels = setupMemoryChannels(
        this->conns_, ctx->memorySemaphores, remoteOutputMemories, inputBufRegMem, nChannelsPerConnection);
    ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] IB hybrid: %zu memoryChannels (intra-node)",
         rank, ctx->memoryChannels.size());

    // PortChannel for inter-node IB connections only (CPU proxy → RDMA)
    ctx->proxyService = std::make_shared<ProxyService>();
    ctx->portChannels = setupPortChannels(
        ctx->proxyService, *comm, this->conns_, remoteOutputMemories, inputBufRegMem);
    ctx->portChannelDeviceHandles = setupPortChannelDeviceHandles(ctx->portChannels);
    ctx->proxyService->startProxy(true);
    INFO(MSCCLPP_COLL, "[alltoallv][rank %d] IB hybrid: %zu portChannels (inter-node), proxy started",
         rank, ctx->portChannels.size());

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

}  // namespace collective
}  // namespace mscclpp
