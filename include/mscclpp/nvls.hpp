// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NVLS_HPP_
#define MSCCLPP_NVLS_HPP_

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/nvls_device.hpp>

namespace mscclpp {

class NvlsConnection {
 public:
  NvlsConnection(size_t bufferSize, int numDevices);
  NvlsConnection(const std::vector<char>& data);
  NvlsConnection() = delete;
  std::vector<char> serialize();

  // Everyone needs to synchronize after creating a NVLS connection before adding devices
  void addDevice();
  void addDevice(int cudaDeviceId);

  struct DeviceMulticastPointer {
   private:
    void* devicePtr_;
    std::shared_ptr<char> mcPtr_;
    size_t bufferSize_;

   public:
    using DeviceHandle = DeviceMulticastPointerDeviceHandle;
    DeviceMulticastPointer(void* devicePtr, std::shared_ptr<char> mcPtr, size_t bufferSize)
        : devicePtr_(devicePtr), mcPtr_(mcPtr), bufferSize_(bufferSize) {}
    DeviceHandle deviceHandle();
    void* getDevicePtr();

    friend class NvlsConnection;
  };

  /// @brief bind the memory allocated via @ref mscclpp::GpuBuffer to the multicast handle. The behavior
  /// is undefined if the devicePtr is not allocated by @ref mscclpp::GpuBuffer.
  /// @param devicePtr The device pointer returned by `mscclpp::GpuBuffer::data()`.
  /// @param size The bytes of the memory to bind to the multicast handle.
  /// @return DeviceMulticastPointer with devicePtr, mcPtr and bufferSize
  DeviceMulticastPointer bindAllocatedMemory(CUdeviceptr devicePtr, size_t size);

  size_t getMultiCastMinGranularity();

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

}  // namespace mscclpp

#endif  // MSCCLPP_NVLS_HPP_
