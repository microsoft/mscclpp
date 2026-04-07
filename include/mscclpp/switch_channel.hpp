// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_HPP_
#define MSCCLPP_SWITCH_CHANNEL_HPP_

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/switch_channel_device.hpp>

namespace mscclpp {

class NvlsConnection;

struct SwitchChannel {
 private:
  void* devicePtr_;
  std::shared_ptr<void> mcPtr_;
  size_t bufferSize_;

 public:
  using DeviceHandle = SwitchChannelDeviceHandle;
  SwitchChannel(void* devicePtr, std::shared_ptr<void> mcPtr, size_t bufferSize)
      : devicePtr_(devicePtr), mcPtr_(mcPtr), bufferSize_(bufferSize) {}
  DeviceHandle deviceHandle() const;
  void* getDevicePtr();

  friend class NvlsConnection;
  friend struct SwitchGroupSemaphore;
};

class NvlsConnection {
 public:
  NvlsConnection(size_t bufferSize, int numDevices);
  NvlsConnection(const std::vector<char>& data);
  NvlsConnection() = delete;
  std::vector<char> serialize();

  /// Bind the memory allocated via mscclpp::GpuBuffer to the multicast handle. The behavior
  /// is undefined if the devicePtr is not allocated by mscclpp::GpuBuffer.
  /// @param devicePtr The device pointer returned by `mscclpp::GpuBuffer::data()`.
  /// @param size The bytes of the memory to bind to the multicast handle.
  /// @return SwitchChannel with devicePtr, mcPtr and bufferSize
  SwitchChannel bindAllocatedMemory(CUdeviceptr devicePtr, size_t size);

 private:
  class Impl;
  std::shared_ptr<Impl> pimpl_;
};

class Communicator;

/// Connect to NVLS on setup.
///
/// This function used to connect to NVLS on setup. NVLS collective using multicast operations to send/recv data.
/// Here we need to put all involved ranks into the collective group.
///
/// @param comm The communicator.
/// @param allRanks The ranks of all processes involved in the collective.
/// @param config The configuration for the local endpoint.
/// @return std::shared_ptr<NvlsConnection> A shared pointer to the NVLS connection.
std::shared_ptr<NvlsConnection> connectNvlsCollective(std::shared_ptr<Communicator> comm, std::vector<int> allRanks,
                                                      size_t bufferSize);

/// A semaphore for O(1) signal/wait synchronization across all devices in a switch (multicast) group.
///
/// Uses `multimem.red` hardware reduction on the NVSwitch multicast address to signal all peers
/// simultaneously with a single operation, instead of O(N) point-to-point signals. Each device
/// atomically increments a shared flag via multicast reduction on signal, and polls its local
/// copy of the flag on wait.
///
/// The flag channel must be a @ref SwitchChannel bound to a buffer used exclusively for the flag.
/// The caller is responsible for keeping the flag channel (and its underlying @ref NvlsConnection
/// and @ref GpuBuffer) alive for the lifetime of this semaphore.
///
/// Example usage:
/// @code
/// auto flagBuffer = mscclpp::GpuBuffer<uint32_t>(1);
/// auto flagChannel = nvlsConn->bindAllocatedMemory(CUdeviceptr(flagBuffer.data()), flagBuffer.bytes());
/// auto semaphore = mscclpp::SwitchGroupSemaphore(flagChannel, numDevices);
/// auto devHandle = semaphore.deviceHandle();
/// // In kernel: devHandle.signal(); devHandle.wait();
/// @endcode
struct SwitchGroupSemaphore {
  using DeviceHandle = SwitchGroupSemaphoreDeviceHandle;

  /// Construct a SwitchGroupSemaphore from a SwitchChannel used as the flag channel.
  /// @param flagChannel A SwitchChannel bound to a buffer used for the flag.
  /// @param numDevices The number of devices in the multicast group.
  SwitchGroupSemaphore(SwitchChannel& flagChannel, int numDevices);

  /// Returns the device-side handle.
  /// @return The device-side handle for use in GPU kernels.
  DeviceHandle deviceHandle() const;

 private:
  void* mcFlag_;
  void* deviceFlag_;
  detail::UniqueGpuPtr<uint32_t> expectedInbound_;
  int numDevices_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_HPP_
