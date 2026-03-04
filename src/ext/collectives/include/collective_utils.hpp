// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_
#define MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_

#include <memory>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>
#include <unordered_map>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#else
#define WARP_SIZE 32
#endif

namespace mscclpp {

namespace collective {
constexpr int NUM_NVLS_CONNECTION = 8;
constexpr int NUM_SEMAPHORES = 64;

constexpr int MAX_NRANKS_PER_NODE = 8;

constexpr int SCRATCH_SIZE = 2 * 1024 * 1024 * 70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB
static bool mscclppDisableChannelCache = env()->disableChannelCache;

std::vector<RegisteredMemory> setupRemoteMemories(std::shared_ptr<Communicator> comm, int rank,
                                                  RegisteredMemory localMemory);

std::vector<MemoryChannel> setupMemoryChannels(
    const std::vector<Connection>& connections,
    const std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    const std::vector<RegisteredMemory>& remoteMemories, RegisteredMemory localMemory, int nChannelsPerConnection);

std::vector<Connection> setupConnections(std::shared_ptr<Communicator> comm);

/// Setup connections with hybrid transport: CudaIpc for intra-node, IB for inter-node.
/// Dynamically detects if all peers are intra-node (single-node case) and falls back to CudaIpc-only.
/// @param comm Communicator
/// @param localGpuIdx Local GPU index within the node (used to select IB device)
/// @return Vector of connections (one per peer)
std::vector<Connection> setupHybridConnections(std::shared_ptr<Communicator> comm, int localGpuIdx);

/// Check if a connection is intra-node (CudaIpc transport).
/// @param conn The connection to check
/// @return true if the connection uses CudaIpc transport
inline bool isIntraNodeConnection(const Connection& conn) {
  return conn.transport() == Transport::CudaIpc;
}

/// Get the IB transport for a given local GPU index.
/// @param localGpuIdx Local GPU index (0-7)
/// @return The corresponding IB transport
Transport getIBTransportForGpu(int localGpuIdx);

/// Setup PortChannels for inter-node connections via ProxyService.
/// Creates PortChannels only for IB connections, with MemoryId-based addressing.
/// @param proxyService The ProxyService managing IB transfers
/// @param comm The communicator
/// @param connections All connections (mixed CudaIpc + IB)
/// @param remoteMemories Remote registered memories (one per peer)
/// @param localMemory Local registered memory
/// @return Vector of PortChannels (only for IB peers, in connection order)
std::vector<PortChannel> setupPortChannels(
    std::shared_ptr<ProxyService> proxyService,
    Communicator& comm,
    const std::vector<Connection>& connections,
    const std::vector<RegisteredMemory>& remoteMemories,
    RegisteredMemory localMemory);

/// Setup PortChannel device handles (GPU-allocated array).
std::shared_ptr<PortChannelDeviceHandle> setupPortChannelDeviceHandles(
    const std::vector<PortChannel>& portChannels);

std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> setupMemorySemaphores(
    std::shared_ptr<Communicator> comm, const std::vector<Connection>& connections, int nChannelsPerConnection);

std::shared_ptr<DeviceHandle<MemoryChannel>> setupMemoryChannelDeviceHandles(
    const std::vector<MemoryChannel>& memoryChannels);

std::vector<std::shared_ptr<NvlsConnection>> setupNvlsConnections(std::shared_ptr<Communicator> comm, size_t size,
                                                                  int numConnections);

std::vector<SwitchChannel> setupNvlsChannels(std::vector<std::shared_ptr<NvlsConnection>> conns, void* buffer,
                                             size_t bufferSize, int nSwitchChannels);

std::shared_ptr<DeviceHandle<SwitchChannel>> setupNvlsChannelDeviceHandles(
    const std::vector<SwitchChannel>& nvlsChannels);

std::vector<BaseMemoryChannel> setupBaseMemoryChannels(
    const std::vector<Connection>& connections,
    const std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>>& memorySemaphores, int nChannelsPerConnection);

std::shared_ptr<DeviceHandle<BaseMemoryChannel>> setupBaseMemoryChannelDeviceHandles(
    const std::vector<BaseMemoryChannel>& baseMemoryChannels);

/// Context holding resources for algorithm execution.
///
/// This struct contains all the channels, semaphores, and memory handles
/// needed for executing a native algorithm. It is created once per unique
/// buffer configuration and cached for reuse.
class AlgorithmCtx {
 public:
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<SwitchChannel> switchChannels;
  std::vector<PortChannel> portChannels;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> hostSemaphores;
  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

}  // namespace collective
}  // namespace mscclpp
#endif  // MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_