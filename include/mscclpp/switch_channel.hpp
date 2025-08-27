// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_HPP_
#define MSCCLPP_SWITCH_CHANNEL_HPP_

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/switch_channel_device.hpp>

namespace mscclpp {

struct NvlsConnection {
  std::vector<std::shared_ptr<Connection>> rootPeerConnections;
  std::shared_ptr<Connection> rootSelfConnection;
  std::shared_ptr<Connection> connection;
};

class SwitchChannel {
 private:
  void* devicePtr_;
  void* mcPtr_;
  size_t bufferSize_;

 public:
  using DeviceHandle = SwitchChannelDeviceHandle;
  SwitchChannel(std::shared_ptr<NvlsConnection> conn, void* data, size_t bytes);
  DeviceHandle deviceHandle() const;
  void* getDevicePtr();
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
