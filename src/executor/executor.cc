// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "executor.hpp"

#include <set>

namespace {
static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
                                         mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
                                         mscclpp::Transport::IB6, mscclpp::Transport::IB7};
}  // namespace

namespace mscclpp {

Executor::Executor(std::shared_ptr<Communicator> comm, const std::unordered_map<int, mscclpp::Connection> connections)
    : impl_(std::make_shared<Impl>(comm, connections)) {}

void Executor::execute(void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize,
                       const ExecutionPlan& plan) {
  ExecutionContext context =
      this->impl_->setupExecutionContext(0, sendbuff, recvBuff, sendBuffSize, recvBuffSize, plan);
  this->impl_->launchKernel(context);
}

Executor::Impl::Impl(std::shared_ptr<Communicator> comm,
                     const std::unordered_map<int, std::shared_ptr<Connection>> connections)
    : comm(comm), connections(connections) {
  this->proxyService = std::make_shared<ProxyService>();
}

ExecutionContext Executor::Impl::setupExecutionContext(int rank, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                                                       size_t recvBufferSize, const ExecutionPlan& plan) {
  ExecutionContextKey key = {sendbuff, recvbuff, sendBufferSize, recvBufferSize, plan.getName()};
  if (this->contexts.find(key) != this->contexts.end()) {
    return this->contexts[key];
  }
  ExecutionContext context;
  size_t scratchBufferSize = plan.getScratchBufferSize(rank, sendBufferSize);
  std::shared_ptr<char> scratchBuffer = allocExtSharedCuda<char>(scratchBufferSize);
  context.scratchBuffer = scratchBuffer;
  context.scratchBufferSize = scratchBufferSize;
  this->setupRegisteredMemories(context, sendbuff, recvbuff, sendBufferSize, recvBufferSize, rank, plan);
  this->setupChannels(context, sendbuff, recvbuff, sendBufferSize, rank, plan);
  return context;
}

void Executor::Impl::setupRegisteredMemories(ExecutionContext& context, void* sendbuff, void* recvbuff,
                                             size_t sendBufferSize, size_t recvBufferSize, int rank,
                                             const ExecutionPlan& plan) {
  int nranksPerNode = plan.nranksPerNode();
  auto getTransportFlags = [&](std::vector<ChannelInfo>& infos, int rank) {
    TransportFlags flags;
    for (ChannelInfo& info : infos) {
      if (info.channelType == ChannelType::SM) {
        flags |= Transport::CudaIpc;
      } else if (info.channelType == ChannelType::PROXY) {
        flags |= IBs[rank % nranksPerNode];
      }
    }
    return flags;
  };
  auto getBufferInfo = [&](BufferType type) {
    switch (type) {
      case BufferType::INPUT:
        return std::make_pair(sendbuff, sendBufferSize);
      case BufferType::OUTPUT:
        return std::make_pair(recvbuff, recvBufferSize);
      case BufferType::SCRATCH:
        return std::make_pair((void*)context.scratchBuffer.get(), context.scratchBufferSize);
      default:
        throw std::runtime_error("Invalid buffer type");
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

  std::vector<BufferType> bufferTypes = plan.getConnectedBufferTypes(rank);
  for (BufferType bufferType : bufferTypes) {
    std::vector<ChannelInfo> channelInfos = plan.getChannelInfos(rank, bufferType);
    TransportFlags transportFlags = getTransportFlags(channelInfos, rank);
    RegisteredMemory memory =
        this->comm->registerMemory(getBufferInfo(bufferType).first, getBufferInfo(bufferType).second, transportFlags);
    std::vector<int> connectedPeers = getConnectedPeers(channelInfos);
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
    for (int peer : connectedPeers) {
      comm->sendMemoryOnSetup(memory, peer, 0);
      remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(peer, 0));
    }
    comm->setup();
    for (int i = 0; i < remoteRegMemoryFutures.size(); i++) {
      context.registeredMemories[{bufferType, connectedPeers[i]}] = std::move(remoteRegMemoryFutures[i].get());
    }
  }
}

void Executor::Impl::setupChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                                   int rank, const ExecutionPlan& plan) {
  const auto channelTypes = {ChannelType::SM, ChannelType::PROXY};
  std::vector<std::shared_ptr<SmDevice2DeviceSemaphore>> smSemaphores;
  std::vector<mscclpp::SemaphoreId> proxySemaphores;
  for (ChannelType channelType : channelTypes) {
    std::vector<ChannelInfo> channelInfos = plan.getChannelInfos(rank, channelType);
    for (ChannelInfo& info : channelInfos) {
      for (int peer : info.connectedPeers) {
        if (channelType == ChannelType::SM) {
          smSemaphores.push_back(std::make_shared<SmDevice2DeviceSemaphore>(*this->comm, this->connections.at(peer)));
        } else if (channelType == ChannelType::PROXY) {
          proxySemaphores.push_back(this->proxyService->buildAndAddSemaphore(*this->comm, this->connections.at(peer)));
        }
      }
    }
  }
  this->comm->setup();
  context.smSemaphores = std::move(smSemaphores);
  context.proxySemaphores = std::move(proxySemaphores);

  auto getBuffer = [&](BufferType type) {
    switch (type) {
      case BufferType::INPUT:
        return sendbuff;
      case BufferType::OUTPUT:
        return recvbuff;
      case BufferType::SCRATCH:
        return (void*)context.scratchBuffer.get();
      default:
        throw std::runtime_error("Invalid buffer type");
    }
  };
  for (ChannelType channelType : channelTypes) {
    std::vector<ChannelInfo> channelInfos = plan.getChannelInfos(rank, channelType);
    int index = 0;
    for (ChannelInfo& info : channelInfos) {
      void* src = getBuffer(info.srcBufferType);
      void* dst = getBuffer(info.dstBufferType);
      TransportFlags transport = context.registeredMemories.begin()->second.transports();
      RegisteredMemory localMemory = this->comm->registerMemory(src, sendBufferSize, transport);
      for (int peer : info.connectedPeers) {
        if (channelType == ChannelType::SM) {
          context.smChannels.emplace_back(smSemaphores[index], context.registeredMemories[{info.dstBufferType, peer}],
                                          src, nullptr);
        } else if (channelType == ChannelType::PROXY) {
          context.proxyChannels.emplace_back(
              this->proxyService->proxyChannel(proxySemaphores[index]),
              this->proxyService->addMemory(context.registeredMemories[{info.dstBufferType, peer}]),
              this->proxyService->addMemory(localMemory));
        }
      }
    }
  }
}

void Executor::Impl::launchKernel(ExecutionContext& context) {
  // Need to change to use flush function and make sure the proxy service will get the latest data.
  this->proxyService->startProxy();
}

}  // namespace mscclpp
