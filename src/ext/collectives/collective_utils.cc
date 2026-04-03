// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "collective_utils.hpp"

#include <algorithm>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>

namespace mscclpp {
namespace collective {
std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(std::shared_ptr<mscclpp::Communicator> comm, int rank,
                                                           mscclpp::RegisteredMemory localMemory) {
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    remoteRegMemoryFutures.push_back(comm->recvMemory(i));
    comm->sendMemory(localMemory, i);
  }
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(), std::back_inserter(remoteMemories),
                 [](const auto& future) { return future.get(); });
  return remoteMemories;
}

std::vector<mscclpp::MemoryChannel> setupMemoryChannels(
    const std::vector<mscclpp::Connection>& connections,
    const std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    const std::vector<mscclpp::RegisteredMemory>& remoteMemories, mscclpp::RegisteredMemory localMemory,
    int nChannelsPerConnection) {
  std::vector<mscclpp::MemoryChannel> channels;
  // Count number of CudaIpc connections for proper dense indexing into memorySemaphores
  size_t nCudaIpcConns = 0;
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid].transport() == mscclpp::Transport::CudaIpc) nCudaIpcConns++;
  }
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    size_t semIdx = 0;
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nCudaIpcConns + semIdx], remoteMemories[cid], localMemory, nullptr);
        semIdx++;
      }
    }
  }
  return channels;
}

std::vector<mscclpp::Connection> setupConnections(std::shared_ptr<mscclpp::Communicator> comm) {
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == comm->bootstrap()->getRank()) continue;
    connectionFutures.push_back(comm->connect(mscclpp::Transport::CudaIpc, i));
  }
  std::vector<mscclpp::Connection> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });
  return connections;
}

// IB device array — GPU index maps to its dedicated IB device
static const mscclpp::Transport IBs[] = {
    mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
    mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7,
};

mscclpp::Transport getIBTransportForGpu(int localGpuIdx) {
  int ibCount = mscclpp::getIBDeviceCount();
  if (ibCount <= 0) {
    throw std::runtime_error("No IB devices available for inter-node communication");
  }
  int idx = localGpuIdx % ibCount;
  return IBs[idx];
}

std::vector<mscclpp::Connection> setupHybridConnections(std::shared_ptr<mscclpp::Communicator> comm,
                                                        int localGpuIdx) {
  int rank = comm->bootstrap()->getRank();
  int worldSize = comm->bootstrap()->getNranks();
  int nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  int thisNode = rank / nRanksPerNode;

  bool hasIB = mscclpp::getIBDeviceCount() > 0;
  mscclpp::Transport ibTransport = hasIB ? getIBTransportForGpu(localGpuIdx) : mscclpp::Transport::CudaIpc;

  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures;
  for (int r = 0; r < worldSize; r++) {
    if (r == rank) continue;
    mscclpp::Transport transport;
    if (r / nRanksPerNode == thisNode) {
      transport = mscclpp::Transport::CudaIpc;
    } else {
      transport = ibTransport;
    }
    connectionFutures.push_back(comm->connect(transport, r));
  }

  std::vector<mscclpp::Connection> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });
  return connections;
}

std::vector<mscclpp::PortChannel> setupPortChannels(
    std::shared_ptr<mscclpp::ProxyService> proxyService,
    mscclpp::Communicator& comm,
    const std::vector<mscclpp::Connection>& connections,
    const std::vector<mscclpp::RegisteredMemory>& remoteMemories,
    mscclpp::RegisteredMemory localMemory) {
  std::vector<mscclpp::PortChannel> channels;
  mscclpp::MemoryId srcMemId = proxyService->addMemory(localMemory);
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid].transport() != mscclpp::Transport::CudaIpc) {
      // IB connection → PortChannel
      mscclpp::SemaphoreId semId = proxyService->buildAndAddSemaphore(comm, connections[cid]);
      mscclpp::MemoryId dstMemId = proxyService->addMemory(remoteMemories[cid]);
      channels.emplace_back(proxyService->portChannel(semId, dstMemId, srcMemId));
    }
  }
  return channels;
}

std::shared_ptr<mscclpp::PortChannelDeviceHandle> setupPortChannelDeviceHandles(
    const std::vector<mscclpp::PortChannel>& portChannels) {
  if (portChannels.empty()) return nullptr;
  std::vector<mscclpp::PortChannelDeviceHandle> handles;
  std::transform(portChannels.begin(), portChannels.end(), std::back_inserter(handles),
                 [](const mscclpp::PortChannel& ch) { return ch.deviceHandle(); });
  auto ptr = mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(handles.size());
  mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(
      ptr.get(), handles.data(), handles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> setupMemorySemaphores(
    std::shared_ptr<mscclpp::Communicator> comm, const std::vector<mscclpp::Connection>& connections,
    int nChannelsPerConnection) {
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        memorySemaphores.emplace_back(
            std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*(comm), connections[cid]));
      }
    }
  }
  return memorySemaphores;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> setupMemoryChannelDeviceHandles(
    const std::vector<mscclpp::MemoryChannel>& memoryChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles;
  std::transform(memoryChannels.begin(), memoryChannels.end(), std::back_inserter(memoryChannelDeviceHandles),
                 [](const mscclpp::MemoryChannel& memoryChannel) { return mscclpp::deviceHandle(memoryChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> ptr =
      mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>(
          memoryChannelDeviceHandles.size());
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>(
      ptr.get(), memoryChannelDeviceHandles.data(), memoryChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

std::vector<std::shared_ptr<mscclpp::NvlsConnection>> setupNvlsConnections(std::shared_ptr<mscclpp::Communicator> comm,
                                                                           size_t size, int numConnections) {
  // for nvls connection
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnections;
  int nRanks = comm->bootstrap()->getNranks();
  std::vector<int> ranks;
  for (int i = 0; i < nRanks; i++) {
    ranks.push_back(i);
  }
  for (int i = 0; i < numConnections; i++) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = mscclpp::connectNvlsCollective(comm, ranks, size);
    nvlsConnections.push_back(nvlsConnection);
  }
  return nvlsConnections;
}

std::vector<mscclpp::SwitchChannel> setupNvlsChannels(std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns,
                                                      void* buffer, size_t bufferSize, int nSwitchChannels) {
  std::vector<mscclpp::SwitchChannel> channels;

  for (int idx = 0; idx < nSwitchChannels; ++idx) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = conns[idx];
    mscclpp::SwitchChannel switchChannel = nvlsConnection->bindAllocatedMemory((CUdeviceptr)buffer, bufferSize);
    channels.push_back(switchChannel);
  }
  return channels;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> setupNvlsChannelDeviceHandles(
    const std::vector<mscclpp::SwitchChannel>& nvlsChannels) {
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> ptr =
      mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(nvlsChannels.size());
  std::vector<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> nvlsChannelDeviceHandles;
  std::transform(nvlsChannels.begin(), nvlsChannels.end(), std::back_inserter(nvlsChannelDeviceHandles),
                 [](const mscclpp::SwitchChannel& nvlsChannel) { return mscclpp::deviceHandle(nvlsChannel); });
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::SwitchChannel>>(
      ptr.get(), nvlsChannelDeviceHandles.data(), nvlsChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

std::vector<mscclpp::BaseMemoryChannel> setupBaseMemoryChannels(
    const std::vector<mscclpp::Connection>& connections,
    const std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    int nChannelsPerConnection) {
  std::vector<mscclpp::BaseMemoryChannel> channels;
  size_t nConnections = connections.size();
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nConnections + cid]);
      }
    }
  }
  return channels;
}

std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> setupBaseMemoryChannelDeviceHandles(
    const std::vector<mscclpp::BaseMemoryChannel>& baseMemoryChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> memoryChannelDeviceHandles;
  std::transform(baseMemoryChannels.begin(), baseMemoryChannels.end(), std::back_inserter(memoryChannelDeviceHandles),
                 [](const mscclpp::BaseMemoryChannel& memoryChannel) { return mscclpp::deviceHandle(memoryChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> ptr =
      mscclpp::detail::gpuCallocShared<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>(
          memoryChannelDeviceHandles.size());
  mscclpp::gpuMemcpy<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>>(
      ptr.get(), memoryChannelDeviceHandles.data(), memoryChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

}  // namespace collective

}  // namespace mscclpp