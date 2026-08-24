// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>

#include "execution_kernel.hpp"
#include "execution_plan.hpp"

namespace mscclpp {

struct ExecutionContextKey {
  void* sendBuff;
  void* recvBuff;
  size_t sendBuffSize;
  size_t recvBuffSize;
  std::string plan;

  bool operator==(const ExecutionContextKey& other) const {
    return sendBuff == other.sendBuff && recvBuff == other.recvBuff && sendBuffSize == other.sendBuffSize &&
           recvBuffSize == other.recvBuffSize && plan == other.plan;
  }
};

std::pair<void*, size_t> getBufferInfo(BufferType type, void* sendbuff, void* recvbuff, void* scratch,
                                       size_t sendBuffSize, size_t recvBuffSize, size_t scratchBuffSize) {
  switch (type) {
    case BufferType::INPUT:
      return std::make_pair(sendbuff, sendBuffSize);
    case BufferType::OUTPUT:
      return std::make_pair(recvbuff, recvBuffSize);
    case BufferType::SCRATCH:
      return std::make_pair(scratch, scratchBuffSize);
    default:
      throw Error("Invalid buffer type", ErrorCode::ExecutorError);
  }
};

struct DeviceExecutionPlanKey {
  size_t inputMessageSize;
  size_t outputMessageSize;
  size_t constSrcOffset;
  size_t constDstOffset;

  bool operator==(const DeviceExecutionPlanKey& other) const {
    return inputMessageSize == other.inputMessageSize && outputMessageSize == other.outputMessageSize &&
           constSrcOffset == other.constSrcOffset && constDstOffset == other.constDstOffset;
  }
};

struct PreparedExecutionKey {
  int rank;
  void* sendBuff;
  void* recvBuff;
  size_t sendBuffSize;
  size_t recvBuffSize;
  int dataType;
  int packetType;
  uintptr_t planIdentity;
  size_t planHash;
  int worldSize;
  int ipcDomainSize;

  bool operator==(const PreparedExecutionKey& other) const {
    return rank == other.rank && sendBuff == other.sendBuff && recvBuff == other.recvBuff &&
           sendBuffSize == other.sendBuffSize && recvBuffSize == other.recvBuffSize && dataType == other.dataType &&
           packetType == other.packetType && planIdentity == other.planIdentity && planHash == other.planHash &&
           worldSize == other.worldSize && ipcDomainSize == other.ipcDomainSize;
  }
};

struct PreparedExecutionKeyHash {
  size_t operator()(const PreparedExecutionKey& key) const {
    size_t seed = 42;
    detail::hashCombine(seed, key.rank);
    detail::hashCombine(seed, key.sendBuff);
    detail::hashCombine(seed, key.recvBuff);
    detail::hashCombine(seed, key.sendBuffSize);
    detail::hashCombine(seed, key.recvBuffSize);
    detail::hashCombine(seed, key.dataType);
    detail::hashCombine(seed, key.packetType);
    detail::hashCombine(seed, key.planIdentity);
    detail::hashCombine(seed, key.planHash);
    detail::hashCombine(seed, key.worldSize);
    detail::hashCombine(seed, key.ipcDomainSize);
    return seed;
  }
};

}  // namespace mscclpp

namespace std {

template <>
struct hash<std::pair<mscclpp::BufferType, int>> {
  std::size_t operator()(const std::pair<mscclpp::BufferType, int>& key) const {
    std::size_t seed = 42;
    mscclpp::detail::hashCombine(seed, static_cast<int>(key.first));
    mscclpp::detail::hashCombine(seed, key.second);
    return seed;
  }
};

template <>
struct hash<mscclpp::ExecutionContextKey> {
  std::size_t operator()(const mscclpp::ExecutionContextKey& key) const {
    size_t seed = 42;
    mscclpp::detail::hashCombine(seed, key.sendBuff);
    mscclpp::detail::hashCombine(seed, key.recvBuff);
    mscclpp::detail::hashCombine(seed, key.sendBuffSize);
    mscclpp::detail::hashCombine(seed, key.recvBuffSize);
    mscclpp::detail::hashCombine(seed, key.plan);
    return seed;
  }
};

template <>
struct hash<mscclpp::DeviceExecutionPlanKey> {
  std::size_t operator()(const mscclpp::DeviceExecutionPlanKey& key) const {
    std::size_t seed = 42;
    mscclpp::detail::hashCombine(seed, key.inputMessageSize);
    mscclpp::detail::hashCombine(seed, key.outputMessageSize);
    mscclpp::detail::hashCombine(seed, key.constSrcOffset);
    mscclpp::detail::hashCombine(seed, key.constDstOffset);
    return seed;
  }
};
}  // namespace std

namespace {
auto hasIBDevices = []() { return mscclpp::getIBDeviceCount() > 0; };

auto useIB = [](int rank1, int rank2, int nranksPerIpcDomain) {
  bool inSameIpcDomain = rank1 / nranksPerIpcDomain == rank2 / nranksPerIpcDomain;
  return hasIBDevices() && !inSameIpcDomain;
};

static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                         mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                         mscclpp::Transport::IB6, mscclpp::Transport::IB7};
}  // namespace

namespace mscclpp {

struct ExecutionContext {
  std::shared_ptr<ProxyService> proxyService;
  std::vector<Connection> connections;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections;
  MemoryId localMemoryIdBegin = MemoryId(0);

  // For registered memories, registeredMemoryAddresses is used for memoryChannel and registeredMemoryIds is used for
  // proxy channel
  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::vector<void*> registeredMemoryAddresses;
  std::vector<mscclpp::MemoryId> registeredMemoryIds;
  // local registered memories to keep resources alive
  std::vector<mscclpp::RegisteredMemory> localRegisteredMemories;

  std::vector<mscclpp::BaseMemoryChannel> memoryChannels;
  std::vector<mscclpp::BasePortChannel> portChannels;
  std::vector<mscclpp::SwitchChannel> nvlsChannels;
  std::unordered_map<DeviceExecutionPlanKey, std::vector<DeviceExecutionPlan>> deviceExecutionPlans;
  std::unordered_map<DeviceExecutionPlanKey, std::shared_ptr<char>> deviceExecutionPlansBuffers;
  std::shared_ptr<char> scratchBuffer;
  std::shared_ptr<char> smemaphores;
  size_t scratchBufferSize;
  uint32_t scratchChunkSize;
  int nthreadsPerBlock;
  DeviceExecutionPlanKey currentDevicePlan;
  bool reuseResources;
  bool doubleScratchBuff;
};

struct Executor::Impl {
  int nranksPerNode;
  int nranksPerIpcDomain;
  int nranks;
  std::shared_ptr<Communicator> comm;
  const size_t defaultScratchBufferSize = (1 << 27);
  std::shared_ptr<char> defaultScratchBuffer;
  std::shared_ptr<ProxyService> proxyService;
  std::unordered_map<ExecutionContextKey, ExecutionContext> contexts;
  std::unordered_map<PreparedExecutionKey, ExecutionContextKey, PreparedExecutionKeyHash> preparedContexts;

  Impl(std::shared_ptr<Communicator> comm, std::shared_ptr<char> defaultScratchBuffer = nullptr)
      : comm(comm), defaultScratchBuffer(defaultScratchBuffer) {
    this->nranksPerNode = comm->bootstrap()->getNranksPerNode();
    this->nranksPerIpcDomain = comm->bootstrap()->getNranksPerIpcDomain();
    this->nranks = comm->bootstrap()->getNranks();
    this->proxyService = std::make_shared<ProxyService>();
    this->proxyService->startProxy(true);
  }
  ~Impl() {
    if (proxyService) proxyService->stopProxy();
  }

  size_t planHash(const ExecutionPlan& plan) const {
    size_t seed = 42;
    detail::hashCombine(seed, plan.impl_->planPath);
    detail::hashCombine(seed, plan.impl_->name);
    detail::hashCombine(seed, plan.impl_->collective);
    detail::hashCombine(seed, plan.impl_->rank);
    detail::hashCombine(seed, plan.impl_->reuseResources);
    detail::hashCombine(seed, plan.impl_->doubleScratchBuffer);
    detail::hashCombine(seed, plan.impl_->nThreadsPerBlock);
    return seed;
  }

  size_t preparedResourceBytes() const {
    size_t bytes = 0;
    for (const auto& prepared : preparedContexts) {
      auto context = contexts.find(prepared.second);
      if (context == contexts.end()) continue;
      bytes += context->second.scratchBufferSize;
      for (const auto& plan : context->second.deviceExecutionPlans)
        bytes += plan.second.size() * sizeof(DeviceExecutionPlan);
    }
    return bytes;
  }

  PreparedExecutionKey preparedKey(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize,
                                   size_t recvBuffSize, DataType dataType, const ExecutionPlan& plan,
                                   PacketType packetType) const {
    return {rank, sendbuff, recvbuff, sendBuffSize, recvBuffSize, static_cast<int>(dataType),
            static_cast<int>(packetType), reinterpret_cast<uintptr_t>(plan.impl_.get()), planHash(plan), nranks,
            nranksPerIpcDomain};
  }

  const ExecutionContext& setupExecutionContext(int rank, void* sendbuff, void* recvbuff, size_t inputMessageSize,
                                                size_t outputMessageSize, size_t constSrcOffset, size_t constDstOffset,
                                                size_t sendMemRange, size_t recvMemRange, const ExecutionPlan& plan,
                                                std::shared_ptr<ProxyService> proxyService, bool forceExactKey = false,
                                                DataType contextDtype = DataType::AUTO) {
    ExecutionContextKey key = {sendbuff, recvbuff, sendMemRange, recvMemRange, plan.impl_->name};
    DeviceExecutionPlanKey devicePlanKey = {inputMessageSize, outputMessageSize, constSrcOffset, constDstOffset};

    // The plan is not related to any specific input/output message size or memory address
    if (plan.impl_->reuseResources && !forceExactKey) {
      key = {nullptr, nullptr, 0, 0, plan.impl_->name};
    } else if (forceExactKey) {
      key.plan += "#prepared:" + std::to_string(inputMessageSize) + ":" + std::to_string(outputMessageSize) + ":" +
                  std::to_string(constSrcOffset) + ":" + std::to_string(constDstOffset) + ":" +
                  std::to_string(static_cast<int>(contextDtype));
    }
    if (this->contexts.find(key) != this->contexts.end()) {
      auto& devicePlans = this->contexts[key].deviceExecutionPlans;
      if (this->contexts[key].currentDevicePlan == devicePlanKey) {
        return this->contexts[key];
      } else if (devicePlans.find(devicePlanKey) != devicePlans.end()) {
        this->contexts[key].currentDevicePlan = devicePlanKey;
        return this->contexts[key];
      }
      plan.impl_->operationsReset();
      plan.impl_->lightLoadExecutionPlan(inputMessageSize, outputMessageSize, constSrcOffset, constDstOffset);
      this->setupDeviceExecutionPlan(this->contexts[key], devicePlanKey, plan);
      this->contexts[key].deviceExecutionPlansBuffers[devicePlanKey] =
          GpuBuffer(devicePlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan)).memory();
      gpuMemcpy(this->contexts[key].deviceExecutionPlansBuffers[devicePlanKey].get(),
                (char*)devicePlans[devicePlanKey].data(),
                devicePlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan), cudaMemcpyHostToDevice);
      this->contexts[key].currentDevicePlan = devicePlanKey;
      return this->contexts[key];
    }

    plan.impl_->reset();
    plan.impl_->loadExecutionPlan(inputMessageSize, outputMessageSize, constSrcOffset, constDstOffset);

    ExecutionContext context;
    context.reuseResources = plan.impl_->reuseResources;
    context.doubleScratchBuff = plan.impl_->doubleScratchBuffer;
    context.proxyService = proxyService;
    context.nthreadsPerBlock = plan.impl_->nThreadsPerBlock;
    this->setupScratchBuffer(context, sendMemRange, recvMemRange, plan);
    this->setupConnections(context, rank, sendMemRange, recvMemRange, context.scratchBufferSize, plan);
    this->setupChannels(context, plan);
    this->setupRegisteredMemories(context, sendbuff, recvbuff, sendMemRange, recvMemRange, rank, plan);
    this->setupNvlsChannels(context, sendbuff, recvbuff, rank, sendMemRange, recvMemRange, context.scratchBufferSize,
                            plan);
    this->setupSemaphores(context, plan);
    this->setupDeviceExecutionPlan(context, devicePlanKey, plan);
    context.deviceExecutionPlansBuffers[devicePlanKey] =
        GpuBuffer(context.deviceExecutionPlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan)).memory();
    gpuMemcpy(context.deviceExecutionPlansBuffers[devicePlanKey].get(),
              (char*)context.deviceExecutionPlans[devicePlanKey].data(),
              context.deviceExecutionPlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan), cudaMemcpyHostToDevice);
    context.currentDevicePlan = devicePlanKey;
    this->contexts.insert({key, context});
    return context;
  }

  TransportFlags getTransportFlags(const BufferInfo& info, int rank) {
    TransportFlags flags;
    for (const ChannelType& type : info.accessChannelTypes) {
      if (type == ChannelType::MEMORY) {
        flags |= Transport::CudaIpc;
      } else if (type == ChannelType::PORT) {
        if (useIB(rank, info.accessRank, this->nranksPerIpcDomain)) {
          flags |= IBs[rank % this->nranksPerNode];
        } else
          flags |= Transport::CudaIpc;
      }
    }
    return flags;
  };

  void setupScratchBuffer(ExecutionContext& context, size_t sendBuffSize, size_t recvBuffSize,
                          const ExecutionPlan& plan) {
    size_t scratchBufferSize = plan.impl_->calScratchBufferSize(std::min(sendBuffSize, plan.impl_->maxMessageSize),
                                                                std::min(recvBuffSize, plan.impl_->maxMessageSize));
    context.scratchChunkSize = plan.impl_->calMaxScratchChunkSize(scratchBufferSize);
    if (plan.impl_->reuseResources) {
      if (this->defaultScratchBuffer == nullptr) {
        this->defaultScratchBuffer = GpuBuffer(this->defaultScratchBufferSize).memory();
      }
      if (scratchBufferSize > this->defaultScratchBufferSize) {
        throw Error("Scratch buffer size (" + std::to_string(scratchBufferSize) +
                        " bytes) exceeds default buffer size (" + std::to_string(this->defaultScratchBufferSize) +
                        " bytes). Consider increasing the default scratch buffer size or disabling resource reuse.",
                    ErrorCode::ExecutorError);
      }
      context.scratchBufferSize = this->defaultScratchBufferSize;
      context.scratchBuffer = this->defaultScratchBuffer;
    } else {
      context.scratchBufferSize = scratchBufferSize;
      context.scratchBuffer = GpuBuffer(scratchBufferSize).memory();
    }
  }

  void setupConnections(ExecutionContext& context, int rank, size_t sendBuffSize, size_t recvBuffSize,
                        size_t scratchBuffSize, const ExecutionPlan& plan) {
    auto getBufferSize = [&](BufferType bufferType) {
      switch (bufferType) {
        case BufferType::INPUT:
          return sendBuffSize;
        case BufferType::OUTPUT:
          return recvBuffSize;
        case BufferType::SCRATCH:
          return scratchBuffSize;
        default:
          throw Error("Invalid buffer type", ErrorCode::ExecutorError);
      }
    };

    // Create one connection (unique QP) per channel entry. Each channel gets its own
    // QP — no shared connections.
    // Use per-peer tag counters so that matched connections between pairs of ranks use
    // the same tag, regardless of the order peers appear in each rank's connected_to list.
    std::unordered_map<int, int> peerTagCounters;
    Transport ibTransport = IBs[rank % this->nranksPerNode];
    std::vector<std::shared_future<Connection>> connFutures;
    for (ChannelType channelType : {ChannelType::MEMORY, ChannelType::PORT}) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(channelType);
      for (const auto& info : channelInfos) {
        for (int peer : info.connectedPeers) {
          Transport transport = channelType == ChannelType::PORT && useIB(rank, peer, this->nranksPerIpcDomain)
                                    ? ibTransport
                                    : Transport::CudaIpc;
          connFutures.push_back(this->comm->connect(transport, peer, peerTagCounters[peer]++));
        }
      }
      channelInfos = plan.impl_->getUnpairedChannelInfos(nranks, channelType);
      for (const auto& info : channelInfos) {
        for (int peer : info.connectedPeers) {
          Transport transport = channelType == ChannelType::PORT && useIB(rank, peer, this->nranksPerIpcDomain)
                                    ? ibTransport
                                    : Transport::CudaIpc;
          connFutures.push_back(this->comm->connect(transport, peer, peerTagCounters[peer]++));
        }
      }
    }

    for (auto& future : connFutures) {
      context.connections.push_back(future.get());
    }

    std::vector<NvlsInfo> nvlsInfos = plan.impl_->nvlsInfos.at(rank);
    for (const NvlsInfo& info : nvlsInfos) {
      std::shared_ptr<NvlsConnection> nvlsConnection =
          mscclpp::connectNvlsCollective(this->comm, info.ranks, getBufferSize(info.bufferType));
      context.nvlsConnections.push_back(nvlsConnection);
    }
  }

  void setupRegisteredMemories(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                               size_t recvBufferSize, int rank, const ExecutionPlan& plan) {
    // Add local src,dst and scratch to registeredMemoryIds
    context.localMemoryIdBegin = context.proxyService->nextMemoryId(3);
    for (auto& bufferType : {BufferType::INPUT, BufferType::OUTPUT, BufferType::SCRATCH}) {
      TransportFlags flags = Transport::CudaIpc;
      if (hasIBDevices()) flags |= IBs[rank % this->nranksPerNode];
      RegisteredMemory localMemory;
      auto bufferInfo = getBufferInfo(bufferType, sendbuff, recvbuff, context.scratchBuffer.get(), sendBufferSize,
                                      recvBufferSize, context.scratchBufferSize);
      if (bufferInfo.second > 0) {
        localMemory = this->comm->registerMemory(bufferInfo.first, bufferInfo.second, flags);
      }
      context.proxyService->addMemory(localMemory);
    }

    for (const auto& buffer : plan.impl_->getLocalBufferToSend()) {
      auto bufferInfo = getBufferInfo(buffer.bufferType, sendbuff, recvbuff, context.scratchBuffer.get(),
                                      sendBufferSize, recvBufferSize, context.scratchBufferSize);
      RegisteredMemory memory =
          this->comm->registerMemory(bufferInfo.first, bufferInfo.second, getTransportFlags(buffer, rank));
      comm->sendMemory(memory, buffer.accessRank);
      context.localRegisteredMemories.emplace_back(std::move(memory));
    }
    for (const auto& bufferInfo : plan.impl_->getRemoteBufferInfos()) {
      std::shared_future<RegisteredMemory> remoteRegMemoryFuture = comm->recvMemory(bufferInfo.rank);
      context.registeredMemories.emplace_back(std::move(remoteRegMemoryFuture.get()));
      for (ChannelType chanType : bufferInfo.accessChannelTypes) {
        if (chanType == ChannelType::MEMORY) {
          context.registeredMemoryAddresses.push_back(context.registeredMemories.back().data());
        } else if (chanType == ChannelType::PORT) {
          context.registeredMemoryIds.push_back(context.proxyService->addMemory(context.registeredMemories.back()));
        }
      }
    }
  }

  void setupChannels(ExecutionContext& context, const ExecutionPlan& plan) {
    const auto channelTypes = {ChannelType::MEMORY, ChannelType::PORT};
    std::vector<std::shared_future<Semaphore>> futureMemorySemaphores;
    std::vector<std::shared_future<Semaphore>> futureProxySemaphores;
    std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
    std::vector<mscclpp::SemaphoreId> proxySemaphores;
    int connIdx = 0;
    auto processChannelInfos = [&](std::vector<ChannelInfo>& channelInfos) {
      for (ChannelInfo& info : channelInfos) {
        for (size_t i = 0; i < info.connectedPeers.size(); i++) {
          auto& connection = context.connections[connIdx++];
          if (info.channelType == ChannelType::MEMORY) {
            futureMemorySemaphores.push_back(this->comm->buildSemaphore(
                connection, this->comm->remoteRankOf(connection), this->comm->tagOf(connection)));
          } else if (info.channelType == ChannelType::PORT) {
            futureProxySemaphores.push_back(this->comm->buildSemaphore(connection, this->comm->remoteRankOf(connection),
                                                                       this->comm->tagOf(connection)));
          }
        }
      }
    };
    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(channelType);
      processChannelInfos(channelInfos);
      // Current semaphore construction requires two-way communication, e.g., to construct a semaphore signaling from
      // rank 0 to rank 1, both rank 0 and rank 1 need to send a message to each other. This PR fixes an executor bug
      // that fails to conduct two-way communication for constructing such one-way semaphores, and instead hangs
      // during the semaphore construction.
      channelInfos = plan.impl_->getUnpairedChannelInfos(nranks, channelType);
      processChannelInfos(channelInfos);
    }

    for (auto sem : futureMemorySemaphores) {
      memorySemaphores.push_back(std::make_shared<MemoryDevice2DeviceSemaphore>(sem.get()));
    }
    for (auto sem : futureProxySemaphores) {
      proxySemaphores.push_back(context.proxyService->addSemaphore(sem.get()));
    }

    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(channelType);
      int index = 0;
      for (ChannelInfo& info : channelInfos) {
        for (size_t i = 0; i < info.connectedPeers.size(); i++) {
          if (channelType == ChannelType::MEMORY) {
            context.memoryChannels.emplace_back(memorySemaphores[index++]);
          } else if (channelType == ChannelType::PORT) {
            context.portChannels.emplace_back(context.proxyService->basePortChannel(proxySemaphores[index++]));
          }
        }
      }
    }
  }

  void setupNvlsChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, int rank, size_t sendBuffSize,
                         size_t recvBuffSize, size_t scratchBuffSize, const ExecutionPlan& plan) {
    std::vector<NvlsInfo> nvlsInfos = plan.impl_->nvlsInfos.at(rank);
    for (size_t i = 0; i < nvlsInfos.size(); i++) {
      std::shared_ptr<NvlsConnection> nvlsConnection = context.nvlsConnections[i];
      NvlsInfo info = nvlsInfos[i];
      auto bufferInfo = getBufferInfo(info.bufferType, sendbuff, recvbuff, context.scratchBuffer.get(), sendBuffSize,
                                      recvBuffSize, scratchBuffSize);
      SwitchChannel switchChannel =
          nvlsConnection->bindAllocatedMemory((CUdeviceptr)bufferInfo.first, bufferInfo.second);
      context.nvlsChannels.push_back(switchChannel);
    }
    this->comm->bootstrap()->barrier();
  }

  void setupSemaphores(ExecutionContext& context, const ExecutionPlan& plan) {
    std::vector<DeviceSemaphore> semaphores;
    for (const SemaphoreInfo& info : plan.impl_->semaphoreInfos) {
      DeviceSemaphore semaphore(info.initValue);
      semaphores.push_back(semaphore);
    }
    context.smemaphores = GpuBuffer(semaphores.size() * sizeof(DeviceSemaphore)).memory();
    gpuMemcpy(context.smemaphores.get(), (char*)semaphores.data(), semaphores.size() * sizeof(DeviceSemaphore),
              cudaMemcpyHostToDevice);
  }

  void setupDeviceExecutionPlan(ExecutionContext& context, const DeviceExecutionPlanKey& key,
                                const ExecutionPlan& plan) {
    std::vector<DeviceExecutionPlan> deviceExecutionPlans;
    for (int threadblock = 0; threadblock < plan.impl_->getThreadblockCount(); threadblock++) {
      DeviceExecutionPlan deviceExecutionPlan = {};
      std::vector<Operation> ops = plan.impl_->getOperations(threadblock);
      deviceExecutionPlan.nOperations = ops.size();
      deviceExecutionPlan.nMemoryChannels = plan.impl_->threadblockMemoryChannels.at(threadblock).size();
      deviceExecutionPlan.nPortChannels = plan.impl_->threadblockPortChannels.at(threadblock).size();
      int chanIndex = 0;
      for (const int index : plan.impl_->threadblockMemoryChannels.at(threadblock)) {
        deviceExecutionPlan.channels.memoryChannels[chanIndex++] = mscclpp::deviceHandle(context.memoryChannels[index]);
      }
      chanIndex = 0;
      for (const int index : plan.impl_->threadblockPortChannels.at(threadblock)) {
        deviceExecutionPlan.channels.portChannels[chanIndex++] = mscclpp::deviceHandle(context.portChannels[index]);
      }
      chanIndex = 0;
      for (const int index : plan.impl_->threadblockNvlsChannels.at(threadblock)) {
        deviceExecutionPlan.channels.nvlsChannels[chanIndex++] = mscclpp::deviceHandle(context.nvlsChannels[index]);
      }
      int memIndex = 0;
      for (const auto& pair : plan.impl_->threadblockMemoryChannelBuffers.at(threadblock)) {
        deviceExecutionPlan.remoteBuffers.memoryChannelBufferPtrs[memIndex] =
            context.registeredMemoryAddresses[pair.first];
        deviceExecutionPlan.remoteBuffers.memoryChannelBufferTypes[memIndex++] = pair.second;
      }
      memIndex = 0;
      for (const auto& pair : plan.impl_->threadblockPortChannelBuffers.at(threadblock)) {
        deviceExecutionPlan.remoteBuffers.portChannelBufferIds[memIndex] = context.registeredMemoryIds[pair.first];
        deviceExecutionPlan.remoteBuffers.portChannelBufferTypes[memIndex++] = pair.second;
      }

      if (ops.size() > MAX_OPERATION) {
        throw Error("Executor plan launching " + std::to_string(ops.size()) +
                        " operations, exceeding device execution plan support (" + std::to_string(MAX_OPERATION) + ")",
                    ErrorCode::ExecutorError);
      }
      for (size_t i = 0; i < ops.size(); i++) {
        deviceExecutionPlan.operations[i] = ops[i];
      }
      deviceExecutionPlans.push_back(deviceExecutionPlan);
    }
    context.deviceExecutionPlans[key] = std::move(deviceExecutionPlans);
  }

  template <typename PacketType>
  void launchKernelHelper(ExecutionContext& context, int rank, void* sendbuff, void* recvbuff, DataType dataType,
                          cudaStream_t stream, uint32_t sharedMemSize, const uint32_t& flag) {
    DeviceExecutionPlanKey key = context.currentDevicePlan;
    int nthreadblocks = context.deviceExecutionPlans[key].size();
    void* scratchBuffer = context.scratchBuffer.get();
    size_t scratchOffset = 0;
    if (context.doubleScratchBuff && (flag & 0x1) == 0) {
      scratchOffset = (context.scratchBufferSize) >> 1;
    }
    if (context.reuseResources) {
      ExecutionKernel::launchKernel<PacketType, true>(
          rank, nthreadblocks, context.nthreadsPerBlock, sendbuff, recvbuff, scratchBuffer, scratchOffset,
          context.scratchChunkSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffers[key].get(),
          (DeviceSemaphore*)context.smemaphores.get(), context.localMemoryIdBegin, sharedMemSize, stream, flag);
    } else {
      ExecutionKernel::launchKernel<PacketType, false>(
          rank, nthreadblocks, context.nthreadsPerBlock, sendbuff, recvbuff, scratchBuffer, scratchOffset,
          context.scratchChunkSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffers[key].get(),
          (DeviceSemaphore*)context.smemaphores.get(), context.localMemoryIdBegin, sharedMemSize, stream, flag);
    }
  }

  void launchKernel(ExecutionContext& context, int rank, void* sendbuff, void* recvbuff, DataType dataType,
                    cudaStream_t stream, PacketType packetType) {
    static uint32_t flag = 0;
#if defined(ENABLE_NPKIT)
#if defined(MSCCLPP_USE_ROCM)
    DeviceExecutionPlanKey key = context.currentDevicePlan;
    int nthreadblocks = context.deviceExecutionPlans[key].size();
    if (nthreadblocks > NPKIT_MAX_NUM_GPU_THREADBLOCKS) {
      throw Error("Executor plan launching " + std::to_string(nthreadblocks) +
                      " thread blocks, exceeding NPKit support (" + std::to_string(NPKIT_MAX_NUM_GPU_THREADBLOCKS) +
                      ")",
                  ErrorCode::ExecutorError);
    }
#endif
    size_t sharedMemSize = sizeof(DeviceExecutionPlan) + NPKIT_SHM_NUM_EVENTS * sizeof(NpKitEvent);
#else
    uint32_t sharedMemSize = sizeof(DeviceExecutionPlan);
#endif
    switch (packetType) {
      case PacketType::LL16:
        launchKernelHelper<LL16Packet>(context, rank, sendbuff, recvbuff, dataType, stream, sharedMemSize, ++flag);
        break;
      case PacketType::LL8:
        launchKernelHelper<LL8Packet>(context, rank, sendbuff, recvbuff, dataType, stream, sharedMemSize, ++flag);
        break;
      default:
        throw Error("Invalid packet type", ErrorCode::ExecutorError);
    }
  }
};

Executor::Executor(std::shared_ptr<Communicator> comm, std::shared_ptr<char> defaultScratchBuffer)
    : impl_(std::make_unique<Impl>(comm, defaultScratchBuffer)) {}

void Executor::execute(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize,
                       [[maybe_unused]] size_t recvBuffSize, DataType dataType, const ExecutionPlan& plan,
                       cudaStream_t stream, PacketType packetType) {
  INFO(LogSubsys::EXEC, "Starting execution with plan: ", plan.name(), ", collective: ", plan.collective());
  size_t sendMemRange, recvMemRange;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendMemRange, (CUdeviceptr)sendbuff));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvMemRange, (CUdeviceptr)recvbuff));
  size_t offsetIn = (char*)sendbuff - (char*)sendBasePtr;
  size_t offsetOut = (char*)recvbuff - (char*)recvBasePtr;

  const ExecutionContext& context = this->impl_->setupExecutionContext(
      rank, (void*)sendBasePtr, (void*)recvBasePtr, sendBuffSize, recvBuffSize, offsetIn, offsetOut, sendMemRange,
      recvMemRange, plan, this->impl_->proxyService, false, dataType);
  this->impl_->launchKernel(context, rank, sendbuff, recvbuff, dataType, stream, packetType);
}

bool Executor::prepare(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize, size_t recvBuffSize,
                       DataType dataType, const ExecutionPlan& plan, PacketType packetType) {
  auto preparedKey = impl_->preparedKey(rank, sendbuff, recvbuff, sendBuffSize, recvBuffSize, dataType, plan, packetType);
  if (impl_->preparedContexts.find(preparedKey) != impl_->preparedContexts.end()) return true;
  if ((sendBuffSize != 0 && sendbuff == nullptr) || (recvBuffSize != 0 && recvbuff == nullptr)) return false;

  CUdeviceptr sendBasePtr = 0, recvBasePtr = 0;
  size_t sendMemRange = 0, recvMemRange = 0;
  if (sendBuffSize != 0) MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendMemRange, (CUdeviceptr)sendbuff));
  if (recvBuffSize != 0) MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvMemRange, (CUdeviceptr)recvbuff));
  size_t offsetIn = sendBuffSize == 0 ? 0 : (char*)sendbuff - (char*)sendBasePtr;
  size_t offsetOut = recvBuffSize == 0 ? 0 : (char*)recvbuff - (char*)recvBasePtr;

  impl_->setupExecutionContext(rank, (void*)sendBasePtr, (void*)recvBasePtr, sendBuffSize, recvBuffSize, offsetIn,
                               offsetOut, sendMemRange, recvMemRange, plan, impl_->proxyService, true, dataType);
  ExecutionContextKey contextKey = {
      (void*)sendBasePtr, (void*)recvBasePtr, sendMemRange, recvMemRange,
      plan.impl_->name + "#prepared:" + std::to_string(sendBuffSize) + ":" + std::to_string(recvBuffSize) + ":" +
          std::to_string(offsetIn) + ":" + std::to_string(offsetOut) + ":" +
          std::to_string(static_cast<int>(dataType))};
  impl_->preparedContexts.emplace(preparedKey, std::move(contextKey));
  return true;
}

bool Executor::executePrepared(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize, size_t recvBuffSize,
                               DataType dataType, const ExecutionPlan& plan, cudaStream_t stream,
                               PacketType packetType) {
  auto key = impl_->preparedKey(rank, sendbuff, recvbuff, sendBuffSize, recvBuffSize, dataType, plan, packetType);
  auto prepared = impl_->preparedContexts.find(key);
  if (prepared == impl_->preparedContexts.end()) return false;
  auto context = impl_->contexts.find(prepared->second);
  if (context == impl_->contexts.end()) return false;
  impl_->launchKernel(context->second, rank, sendbuff, recvbuff, dataType, stream, packetType);
  return true;
}

bool Executor::executeSolePrepared(int rank, void* sendbuff, void* recvbuff, DataType dataType, cudaStream_t stream,
                                   PacketType packetType) {
  if (impl_->preparedContexts.size() != 1) return false;
  auto prepared = impl_->preparedContexts.begin();
  auto context = impl_->contexts.find(prepared->second);
  if (context == impl_->contexts.end()) return false;
  impl_->launchKernel(context->second, rank, sendbuff, recvbuff, dataType, stream, packetType);
  return true;
}

bool Executor::releasePrepared(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize, size_t recvBuffSize,
                               DataType dataType, const ExecutionPlan& plan, PacketType packetType) {
  auto key = impl_->preparedKey(rank, sendbuff, recvbuff, sendBuffSize, recvBuffSize, dataType, plan, packetType);
  auto prepared = impl_->preparedContexts.find(key);
  if (prepared == impl_->preparedContexts.end()) return false;
  impl_->contexts.erase(prepared->second);
  impl_->preparedContexts.erase(prepared);
  return true;
}

void Executor::resetPrepared() {
  for (const auto& entry : impl_->preparedContexts) impl_->contexts.erase(entry.second);
  impl_->preparedContexts.clear();
}

size_t Executor::preparedResourceBytes() const { return impl_->preparedResourceBytes(); }

void Executor::reset() {
  this->impl_->proxyService->stopProxy();
  this->impl_->contexts.clear();
  this->impl_->proxyService = std::make_shared<ProxyService>();
  this->impl_->proxyService->startProxy(true);
}

Executor::~Executor() = default;

}  // namespace mscclpp
