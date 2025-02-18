// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/port_channel.hpp>
#include <set>

#include "debug.h"
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

void* getBuffer(BufferType type, void* sendbuff, void* recvbuff, void* scratch) {
  switch (type) {
    case BufferType::INPUT:
      return sendbuff;
    case BufferType::OUTPUT:
      return recvbuff;
    case BufferType::SCRATCH:
      return scratch;
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

}  // namespace mscclpp

namespace std {

// Refer https://www.boost.org/doc/libs/1_86_0/libs/container_hash/doc/html/hash.html#combine
template <typename T>
inline void hash_combine(std::size_t& seed, const T& value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct hash<std::pair<mscclpp::BufferType, int>> {
  std::size_t operator()(const std::pair<mscclpp::BufferType, int>& key) const {
    std::size_t seed = 0;
    hash_combine(seed, static_cast<int>(key.first));
    hash_combine(seed, key.second);
    return seed;
  }
};

template <>
struct hash<mscclpp::ExecutionContextKey> {
  std::size_t operator()(const mscclpp::ExecutionContextKey& key) const {
    size_t seed = 0;
    hash_combine(seed, key.sendBuff);
    hash_combine(seed, key.recvBuff);
    hash_combine(seed, key.sendBuffSize);
    hash_combine(seed, key.recvBuffSize);
    hash_combine(seed, key.plan);
    return seed;
  }
};

template <>
struct hash<mscclpp::DeviceExecutionPlanKey> {
  std::size_t operator()(const mscclpp::DeviceExecutionPlanKey& key) const {
    std::size_t seed = 0;
    hash_combine(seed, key.inputMessageSize);
    hash_combine(seed, key.outputMessageSize);
    hash_combine(seed, key.constSrcOffset);
    hash_combine(seed, key.constDstOffset);
    return seed;
  }
};
}  // namespace std

namespace {
auto inSameNode = [](int rank1, int rank2, int nranksPerNode) {
  return rank1 / nranksPerNode == rank2 / nranksPerNode;
};

static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                         mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                         mscclpp::Transport::IB6, mscclpp::Transport::IB7};
}  // namespace

namespace mscclpp {

struct ExecutionContext {
  std::shared_ptr<ProxyService> proxyService;
  std::unordered_map<int, std::shared_ptr<Connection>> connections;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections;
  std::unordered_map<std::pair<BufferType, int>, mscclpp::RegisteredMemory> registeredMemories;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<mscclpp::SemaphoreId> proxySemaphores;
  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::vector<mscclpp::PortChannel> portChannels;
  std::vector<mscclpp::NvlsConnection::DeviceMulticastPointer> nvlsChannels;
  std::unordered_map<DeviceExecutionPlanKey, std::vector<DeviceExecutionPlan>> deviceExecutionPlans;
  std::unordered_map<DeviceExecutionPlanKey, std::shared_ptr<char>> deviceExecutionPlansBuffers;
  std::shared_ptr<char> scratchBuffer;
  size_t scratchBufferSize;
  int nthreadsPerBlock;
  DeviceExecutionPlanKey currentDevicePlan;
};

struct Executor::Impl {
  int nranksPerNode;
  int nranks;
  std::shared_ptr<Communicator> comm;
  std::unordered_map<ExecutionContextKey, ExecutionContext> contexts;

  Impl(std::shared_ptr<Communicator> comm) : comm(comm) {
    this->nranksPerNode = comm->bootstrap()->getNranksPerNode();
    this->nranks = comm->bootstrap()->getNranks();
  }
  ~Impl() = default;

  ExecutionContext setupExecutionContext(int rank, void* sendbuff, void* recvbuff, size_t inputMessageSize,
                                         size_t outputMessageSize, size_t constSrcOffset, size_t constDstOffset,
                                         size_t sendMemRange, size_t recvMemRange, const ExecutionPlan& plan) {
    ExecutionContextKey key = {sendbuff, recvbuff, sendMemRange, recvMemRange, plan.impl_->name};
    DeviceExecutionPlanKey devicePlanKey = {inputMessageSize, outputMessageSize, constSrcOffset, constDstOffset};
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
      this->setupDeviceExecutionPlan(this->contexts[key], devicePlanKey, rank, plan);
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
    size_t maxScratchBufferSize = plan.impl_->getMaxScratchBufferSize(rank);
    size_t scratchBufferSize =
        std::min(plan.impl_->getScratchBufferSize(rank, sendMemRange, recvMemRange), maxScratchBufferSize);
    std::shared_ptr<char> scratchBuffer = GpuBuffer(scratchBufferSize).memory();
    context.scratchBuffer = scratchBuffer;
    context.scratchBufferSize = scratchBufferSize;
    context.proxyService = std::make_shared<ProxyService>();
    context.nthreadsPerBlock = plan.impl_->getNThreadsPerBlock();
    this->setupConnections(context, rank, plan, sendMemRange, recvMemRange);
    this->setupRegisteredMemories(context, sendbuff, recvbuff, sendMemRange, recvMemRange, rank, plan);
    this->setupChannels(context, sendbuff, recvbuff, sendMemRange, recvMemRange, rank, plan);
    this->setupNvlsChannels(context, sendbuff, recvbuff, sendMemRange, recvMemRange, rank, plan);
    this->setupDeviceExecutionPlan(context, devicePlanKey, rank, plan);
    context.deviceExecutionPlansBuffers[devicePlanKey] =
        GpuBuffer(context.deviceExecutionPlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan)).memory();
    gpuMemcpy(context.deviceExecutionPlansBuffers[devicePlanKey].get(),
              (char*)context.deviceExecutionPlans[devicePlanKey].data(),
              context.deviceExecutionPlans[devicePlanKey].size() * sizeof(DeviceExecutionPlan), cudaMemcpyHostToDevice);
    context.currentDevicePlan = devicePlanKey;
    context.proxyService->startProxy();
    this->contexts.insert({key, context});
    return context;
  }

  TransportFlags getTransportFlags(std::vector<ChannelInfo>& infos, int rank) {
    TransportFlags flags;
    for (ChannelInfo& info : infos) {
      if (info.channelType == ChannelType::MEMORY) {
        flags |= Transport::CudaIpc;
      } else if (info.channelType == ChannelType::PORT) {
        for (int peer : info.connectedPeers) {
          if (!inSameNode(rank, peer, this->nranksPerNode)) {
            flags |= IBs[rank % this->nranksPerNode];
          } else
            flags |= Transport::CudaIpc;
        }
      }
    }
    return flags;
  };

  void setupConnections(ExecutionContext& context, int rank, const ExecutionPlan& plan, size_t sendBufferSize,
                        size_t recvBufferSize) {
    std::vector<int> connectedPeers = plan.impl_->getConnectedPeers(rank);
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
    for (int peer : connectedPeers) {
      Transport transport =
          inSameNode(rank, peer, this->nranksPerNode) ? Transport::CudaIpc : IBs[rank % this->nranksPerNode];
      connectionFutures.push_back(this->comm->connectOnSetup(peer, 0, transport));
    }
    this->comm->setup();
    for (size_t i = 0; i < connectionFutures.size(); i++) {
      context.connections[connectedPeers[i]] = connectionFutures[i].get();
    }

    std::vector<NvlsInfo> nvlsInfos = plan.impl_->getNvlsInfos(rank, sendBufferSize, recvBufferSize);
    for (const NvlsInfo& info : nvlsInfos) {
      std::shared_ptr<NvlsConnection> nvlsConnection =
          mscclpp::connectNvlsCollective(this->comm, info.ranks, info.bufferSize);
      context.nvlsConnections.push_back(nvlsConnection);
    }
  }

  void setupRegisteredMemories(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                               size_t recvBufferSize, int rank, const ExecutionPlan& plan) {
    auto getBufferInfo = [&](BufferType type) {
      switch (type) {
        case BufferType::INPUT:
          return std::make_pair(sendbuff, sendBufferSize);
        case BufferType::OUTPUT:
          return std::make_pair(recvbuff, recvBufferSize);
        case BufferType::SCRATCH:
          return std::make_pair((void*)context.scratchBuffer.get(), context.scratchBufferSize);
        default:
          throw Error("Invalid buffer type", ErrorCode::ExecutorError);
      }
    };
    auto getConnectedPeers = [&](std::vector<ChannelInfo>& infos) {
      std::set<int> peers;
      for (ChannelInfo& info : infos) {
        for (int peer : info.connectedPeers) {
          peers.insert(peer);
        }
      }
      return std::vector<int>(peers.begin(), peers.end());
    };

    std::vector<BufferType> bufferTypes = plan.impl_->getConnectedBufferTypes(rank);
    for (BufferType bufferType : bufferTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfosByDstRank(rank, bufferType);
      TransportFlags transportFlags = getTransportFlags(channelInfos, rank);
      RegisteredMemory memory =
          this->comm->registerMemory(getBufferInfo(bufferType).first, getBufferInfo(bufferType).second, transportFlags);
      std::vector<int> connectedPeers = getConnectedPeers(channelInfos);
      std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
      for (int peer : connectedPeers) {
        comm->sendMemoryOnSetup(memory, peer, 0);
      }
      channelInfos = plan.impl_->getChannelInfos(rank, bufferType);
      connectedPeers = getConnectedPeers(channelInfos);
      for (int peer : connectedPeers) {
        remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(peer, 0));
      }
      comm->setup();
      for (size_t i = 0; i < remoteRegMemoryFutures.size(); i++) {
        context.registeredMemories[{bufferType, connectedPeers[i]}] = std::move(remoteRegMemoryFutures[i].get());
      }
    }
  }

  void setupChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                     size_t recvBufferSize, int rank, const ExecutionPlan& plan) {
    const auto channelTypes = {ChannelType::MEMORY, ChannelType::PORT};
    std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
    std::vector<mscclpp::SemaphoreId> proxySemaphores;
    auto processChannelInfos = [&](std::vector<ChannelInfo>& channelInfos) {
      for (ChannelInfo& info : channelInfos) {
        for (int peer : info.connectedPeers) {
          if (info.channelType == ChannelType::MEMORY) {
            memorySemaphores.push_back(
                std::make_shared<MemoryDevice2DeviceSemaphore>(*this->comm, context.connections.at(peer)));
          } else if (info.channelType == ChannelType::PORT) {
            proxySemaphores.push_back(
                context.proxyService->buildAndAddSemaphore(*this->comm, context.connections.at(peer)));
          }
        }
      }
    };
    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(rank, channelType);
      processChannelInfos(channelInfos);
      // Current semaphore construction requires two-way communication, e.g., to construct a semaphore signaling from
      // rank 0 to rank 1, both rank 0 and rank 1 need to send a message to each other. This PR fixes an executor bug
      // that fails to conduct two-way communication for constructing such one-way semaphores, and instead hangs
      // during the semaphore construction. In the future, we may need to change the implementation to construct
      // semaphore via one-way communication.
      channelInfos = plan.impl_->getUnpairedChannelInfos(rank, nranks, channelType);
      processChannelInfos(channelInfos);
    }
    this->comm->setup();
    context.memorySemaphores = std::move(memorySemaphores);
    context.proxySemaphores = std::move(proxySemaphores);

    auto getBufferSize = [&](BufferType type) {
      switch (type) {
        case BufferType::INPUT:
          return sendBufferSize;
        case BufferType::OUTPUT:
          return recvBufferSize;
        case BufferType::SCRATCH:
          return context.scratchBufferSize;
        default:
          throw Error("Invalid buffer type", ErrorCode::ExecutorError);
      }
    };

    for (ChannelType channelType : channelTypes) {
      std::vector<ChannelInfo> channelInfos = plan.impl_->getChannelInfos(rank, channelType);
      int index = 0;
      for (ChannelInfo& info : channelInfos) {
        void* src = getBuffer(info.srcBufferType, sendbuff, recvbuff, context.scratchBuffer.get());
        size_t bufferSize = getBufferSize(info.srcBufferType);
        TransportFlags transport = getTransportFlags(channelInfos, rank);
        RegisteredMemory localMemory = this->comm->registerMemory(src, bufferSize, transport);
        for (int peer : info.connectedPeers) {
          if (channelType == ChannelType::MEMORY) {
            context.memoryChannels.emplace_back(context.memorySemaphores[index++],
                                                context.registeredMemories[{info.dstBufferType, peer}], src, nullptr);
          } else if (channelType == ChannelType::PORT) {
            context.portChannels.emplace_back(context.proxyService->portChannel(
                context.proxySemaphores[index++],
                context.proxyService->addMemory(context.registeredMemories[{info.dstBufferType, peer}]),
                context.proxyService->addMemory(localMemory)));
          }
        }
      }
    }
  }

  void setupNvlsChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                         size_t recvBufferSize, int rank, const ExecutionPlan& plan) {
    std::vector<NvlsInfo> nvlsInfos = plan.impl_->getNvlsInfos(rank, sendBufferSize, recvBufferSize);
    for (size_t i = 0; i < nvlsInfos.size(); i++) {
      std::shared_ptr<NvlsConnection> nvlsConnection = context.nvlsConnections[i];
      NvlsInfo info = nvlsInfos[i];
      void* buffer = getBuffer(info.bufferType, sendbuff, recvbuff, context.scratchBuffer.get());
      NvlsConnection::DeviceMulticastPointer deviceMulticastPointer =
          nvlsConnection->bindAllocatedMemory((CUdeviceptr)buffer, info.bufferSize);
      context.nvlsChannels.push_back(deviceMulticastPointer);
    }
  }

  void setupDeviceExecutionPlan(ExecutionContext& context, const DeviceExecutionPlanKey& key, int rank,
                                const ExecutionPlan& plan) {
    std::vector<DeviceExecutionPlan> deviceExecutionPlans;
    for (int threadblock = 0; threadblock < plan.impl_->getThreadblockCount(rank); threadblock++) {
      DeviceExecutionPlan deviceExecutionPlan = {};
      std::vector<Operation> ops = plan.impl_->getOperations(rank, threadblock);
      deviceExecutionPlan.nOperations = ops.size();
      deviceExecutionPlan.nMemoryChannels = plan.impl_->threadblockMemoryChannelMap.at(rank).at(threadblock).size();
      deviceExecutionPlan.nPortChannels = plan.impl_->threadblockPortChannelMap.at(rank).at(threadblock).size();
      int chanIndex = 0;
      for (const auto& [index, _] : plan.impl_->threadblockMemoryChannelMap.at(rank).at(threadblock)) {
        deviceExecutionPlan.channels.memoryChannels[chanIndex++] = mscclpp::deviceHandle(context.memoryChannels[index]);
      }
      chanIndex = 0;
      for (const auto& [index, _] : plan.impl_->threadblockPortChannelMap.at(rank).at(threadblock)) {
        deviceExecutionPlan.channels.portChannels[chanIndex++] = mscclpp::deviceHandle(context.portChannels[index]);
      }
      chanIndex = 0;
      for (const auto& [index, _] : plan.impl_->threadblockNvlsChannelMap.at(rank).at(threadblock)) {
        deviceExecutionPlan.channels.nvlsChannels[chanIndex++] = mscclpp::deviceHandle(context.nvlsChannels[index]);
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

  void launchKernel(ExecutionContext& context, int rank, void* sendbuff, void* recvbuff, DataType dataType,
                    cudaStream_t stream, PacketType packetType) {
    static uint32_t flag = 0;
    DeviceExecutionPlanKey key = context.currentDevicePlan;
    int nthreadblocks = context.deviceExecutionPlans[key].size();
#if defined(ENABLE_NPKIT)
#if defined(__HIP_PLATFORM_AMD__)
    if (nthreadblocks > NPKIT_MAX_NUM_GPU_THREADBLOCKS) {
      throw Error("Executor plan launching " + std::to_string(nthreadblocks) +
                      " thread blocks, exceeding NPKit support (" + std::to_string(NPKIT_MAX_NUM_GPU_THREADBLOCKS) +
                      ")",
                  ErrorCode::ExecutorError);
    }
#endif
    size_t sharedMemSize = sizeof(DeviceExecutionPlan) + NPKIT_SHM_NUM_EVENTS * sizeof(NpKitEvent);
#else
    size_t sharedMemSize = sizeof(DeviceExecutionPlan);
#endif
    switch (packetType) {
      case PacketType::LL16:
        ExecutionKernel::launchKernel<LL16Packet>(
            rank, nthreadblocks, context.nthreadsPerBlock, sendbuff, recvbuff, (void*)context.scratchBuffer.get(),
            context.scratchBufferSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffers[key].get(),
            sharedMemSize, stream, ++flag);
        break;
      case PacketType::LL8:
        ExecutionKernel::launchKernel<LL8Packet>(
            rank, nthreadblocks, context.nthreadsPerBlock, sendbuff, recvbuff, (void*)context.scratchBuffer.get(),
            context.scratchBufferSize, dataType, (DeviceExecutionPlan*)context.deviceExecutionPlansBuffers[key].get(),
            sharedMemSize, stream, ++flag);
        break;
      default:
        throw Error("Invalid packet type", ErrorCode::ExecutorError);
    }
  }
};

Executor::Executor(std::shared_ptr<Communicator> comm) : impl_(std::make_unique<Impl>(comm)) {}

void Executor::execute(int rank, void* sendbuff, void* recvbuff, size_t sendBuffSize,
                       [[maybe_unused]] size_t recvBuffSize, DataType dataType, const ExecutionPlan& plan,
                       cudaStream_t stream, PacketType packetType) {
  INFO(MSCCLPP_EXECUTOR, "Starting execution with plan: %s, collective: %s", plan.name().c_str(),
       plan.collective().c_str());
  size_t sendMemRange, recvMemRange;
  CUdeviceptr sendBasePtr, recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendMemRange, (CUdeviceptr)sendbuff));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvMemRange, (CUdeviceptr)recvbuff));
  size_t offsetIn = (char*)sendbuff - (char*)sendBasePtr;
  size_t offsetOut = (char*)recvbuff - (char*)recvBasePtr;

  ExecutionContext context =
      this->impl_->setupExecutionContext(rank, (void*)sendBasePtr, (void*)recvBasePtr, sendBuffSize, recvBuffSize,
                                         offsetIn, offsetOut, sendMemRange, recvMemRange, plan);
  this->impl_->launchKernel(context, rank, sendbuff, recvbuff, dataType, stream, packetType);
}

Executor::~Executor() = default;

}  // namespace mscclpp
