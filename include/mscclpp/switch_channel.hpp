// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_HPP_
#define MSCCLPP_SWITCH_CHANNEL_HPP_

#include <cstdint>
#include <memory>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/switch_channel_device.hpp>

namespace mscclpp {

class NvlsConnection;

struct SwitchChannel {
 private:
  void* devicePtr_;
  std::shared_ptr<void> mcPtr_;
  size_t bufferSize_;
  // Barrier semaphore inherited from the owning NvlsConnection (see NvlsConnection::bindAllocatedMemory).
  // Its pointers are null if the connection was created without barrier support.
  SwitchDevice2DeviceSemaphoreDeviceHandle barrier_ = {};

 public:
  using DeviceHandle = SwitchChannelDeviceHandle;
  SwitchChannel(void* devicePtr, std::shared_ptr<void> mcPtr, size_t bufferSize)
      : devicePtr_(devicePtr), mcPtr_(mcPtr), bufferSize_(bufferSize) {}
  DeviceHandle deviceHandle() const;
  void* getDevicePtr();

  friend class NvlsConnection;
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

  /// Attach a device-side barrier resource shared by all SwitchChannels created from this
  /// connection. After this call, `SwitchChannel::deviceHandle().barrier()` can synchronize all
  /// ranks in the multicast group without a separate mesh of memory-channel semaphores. This is set
  /// up automatically by `connectNvlsCollective`; it is an internal setup hook and is not intended
  /// to be called directly.
  /// @param barrierBuffer Storage for the barrier flag (kept alive); element 0 is the shared arrival
  /// counter, element 1 is this rank's generation counter.
  /// @param barrierChannel The bound barrier channel (kept alive for its multicast pointer).
  /// @param nRanks Number of ranks participating in the multicast group.
  void attachBarrier(std::shared_ptr<void> barrierBuffer, std::shared_ptr<SwitchChannel> barrierChannel, int nRanks);

 private:
  class Impl;
  std::shared_ptr<Impl> pimpl_;

  // Barrier resources, owned by this connection and shared by every SwitchChannel it creates. The
  // flag is carved from this connection's own multicast region, so no auxiliary connection is kept.
  std::shared_ptr<void> barrierBuffer_;
  std::shared_ptr<SwitchChannel> barrierChannel_;
  SwitchDevice2DeviceSemaphoreDeviceHandle barrierSem_ = {};
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

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_HPP_
