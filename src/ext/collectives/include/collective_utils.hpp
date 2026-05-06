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
// Sized to cover MAX_IPC_DOMAIN_NRANKS-scale allreduce algos whose device-side
// semaphore indices grow as O(ipcDomainNranks) (e.g. nvls_block_pipeline uses
// up to ~5 * ipcDomainNranks entries).
constexpr int NUM_SEMAPHORES = 512;

// Upper bound on the number of NVLink-reachable ranks that participate in a
// single collective. Sized to cover Multi-Node NVLink (MNNVL) domains up to
// GB200 NVL72 (72 GPUs sharing one NVLink fabric). Drives compile-time sizing
// of shared-memory channel arrays in the allreduce/allgather kernels.
constexpr int MAX_IPC_DOMAIN_NRANKS = 72;

constexpr int SCRATCH_SIZE = 2 * 1024 * 1024 * 70;  // double buffer * 35 thread-blocks * 8 ranks * 256KB = 70MB

std::vector<RegisteredMemory> setupRemoteMemories(std::shared_ptr<Communicator> comm, int rank,
                                                  RegisteredMemory localMemory);

std::vector<MemoryChannel> setupMemoryChannels(
    const std::vector<Connection>& connections,
    const std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>>& memorySemaphores,
    const std::vector<RegisteredMemory>& remoteMemories, RegisteredMemory localMemory, int nChannelsPerConnection);

std::vector<Connection> setupConnections(std::shared_ptr<Communicator> comm);
std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> setupMemorySemaphores(
    std::shared_ptr<Communicator> comm, const std::vector<Connection>& connections, int nChannelsPerConnection);

/// Number of ranks that participate in the same GPU-IPC-reachable peer group (e.g. a single host or
/// a Multi-Node NVLink fabric, or an AMD XGMI domain). Returns the value of `MSCCLPP_IPC_DOMAIN_NRANKS`
/// if set to a positive value; otherwise falls back to `bootstrap->getNranksPerNode()`. This is
/// intentionally independent of `Bootstrap::getNranksPerNode()` so that algorithms can opt in to
/// MNNVL-like behavior without changing the meaning of bootstrap-level APIs.
int getIpcDomainNranks(std::shared_ptr<Communicator> comm);

/// Validates that the IPC domain spans the whole communicator and that the local rank fits within
/// the supported `[2, MAX_IPC_DOMAIN_NRANKS]` range. Used by NVLS allreduce algorithms whose
/// multicast group spans the whole communicator. Returns the validated `ipcDomainNranks`; throws
/// `Error(InvalidUsage)` on violation.
int validateIpcDomainSpansWorld(std::shared_ptr<Communicator> comm);

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
  int ipcDomainNranks;

  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<SwitchChannel> switchChannels;
  std::vector<PortChannel> portChannels;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> hostSemaphores;
  std::shared_ptr<void*> remoteMemoryHandles;
  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

}  // namespace collective
}  // namespace mscclpp
#endif  // MSCCLPP_EXT_COLLECTIVE_UTILS_HPP_
