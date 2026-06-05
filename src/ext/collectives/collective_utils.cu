// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "collective_utils.hpp"

#include <algorithm>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/switch_channel.hpp>

namespace mscclpp {
namespace collective {

namespace {

#if !defined(MSCCLPP_DEVICE_HIP)
__global__ void fp8NvlsSupportProbeKernel(int* supported) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
  *supported = 1;
#else
  *supported = 0;
#endif
}

bool detectFp8NvlsSupport() {
  AvoidCudaGraphCaptureGuard cgcGuard;
  auto supportedDevice = mscclpp::detail::gpuCallocUnique<int>();
  int supportedHost = 0;
  auto stream = gpuStreamPool()->getStream();

  fp8NvlsSupportProbeKernel<<<1, 1, 0, stream>>>(supportedDevice.get());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return false;
  }

  MSCCLPP_CUDATHROW(cudaMemcpyAsync(&supportedHost, supportedDevice.get(), sizeof(supportedHost),
                                    cudaMemcpyDeviceToHost, stream));
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return supportedHost != 0;
}
#endif

}  // namespace

bool isFp8DataType(DataType dtype) {
  return dtype == DataType::FLOAT8_E4M3FN || dtype == DataType::FLOAT8_E4M3FNUZ ||
         dtype == DataType::FLOAT8_E5M2 || dtype == DataType::FLOAT8_E5M2FNUZ ||
         dtype == DataType::FLOAT8_E4M3B15;
}

bool isNativeFp8DataType(DataType dtype) {
#if defined(__FP8_TYPES_EXIST__)
#if defined(__FP8_E4M3_IS_FNUZ__)
  if (dtype == DataType::FLOAT8_E4M3FNUZ) {
    return true;
  }
#else
  if (dtype == DataType::FLOAT8_E4M3FN) {
    return true;
  }
#endif
#if defined(__FP8_E5M2_IS_FNUZ__)
  if (dtype == DataType::FLOAT8_E5M2FNUZ) {
    return true;
  }
#else
  if (dtype == DataType::FLOAT8_E5M2) {
    return true;
  }
#endif
#endif
  return false;
}

bool isFp8NvlsSupported() {
#if defined(MSCCLPP_DEVICE_HIP)
  return false;
#else
  static const bool supported = detectFp8NvlsSupport();
  return supported;
#endif
}

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
  size_t nConnections = connections.size();
  for (int idx = 0; idx < nChannelsPerConnection; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (connections[cid].transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(memorySemaphores[idx * nConnections + cid], remoteMemories[cid], localMemory, nullptr);
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

std::vector<mscclpp::SwitchChannel> setupNvlsChannels(std::shared_ptr<mscclpp::Communicator> comm,
                                                      std::vector<std::shared_ptr<mscclpp::NvlsConnection>> conns,
                                                      void* buffer, size_t bufferSize, int nSwitchChannels) {
  std::vector<mscclpp::SwitchChannel> channels;

  for (int idx = 0; idx < nSwitchChannels; ++idx) {
    std::shared_ptr<mscclpp::NvlsConnection> nvlsConnection = conns[idx];
    mscclpp::SwitchChannel switchChannel = nvlsConnection->bindAllocatedMemory((CUdeviceptr)buffer, bufferSize);
    channels.push_back(switchChannel);
  }
  // Synchronize to make sure all ranks have their NVLS channels set up before any rank starts using them.
  comm->bootstrap()->barrier();
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
