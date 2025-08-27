// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <mscclpp/core.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "gpu_ipc_mem.hpp"
#include "serialization.hpp"

namespace mscclpp {

SwitchChannel::SwitchChannel(std::shared_ptr<NvlsConnection> nvlsConn, void* data, size_t bytes) {
  if (!nvlsConn) {
    std::abort();
  }
  CudaIpcConnection* connection = dynamic_cast<CudaIpcConnection*>(nvlsConn->connection.get());
  if (!connection) {
    throw std::runtime_error("Invalid connection type");
  }
  if (!(connection->nvlsMem_)) {
    std::abort();
  }
  (void)connection->nvlsMem_->mapMulticast(connection->nvlsNumDevs_, CUdeviceptr(data), bytes);
  devicePtr_ = data ? data : connection->nvlsMem_->multicastBuffer();
  if (!devicePtr_) {
    std::abort();
  }
  mcPtr_ = connection->nvlsMem_->data();
  bufferSize_ = connection->nvlsMem_->size();
}

SwitchChannel::DeviceHandle SwitchChannel::deviceHandle() const {
  SwitchChannel::DeviceHandle device;
  device.devicePtr = devicePtr_;
  device.mcPtr = mcPtr_;
  device.bufferSize = bufferSize_;
  return device;
};

void* SwitchChannel::getDevicePtr() { return devicePtr_; };

MSCCLPP_API_CPP std::shared_ptr<NvlsConnection> connectNvlsCollective(std::shared_ptr<Communicator> comm,
                                                                      std::vector<int> allRanks, size_t bufferSize) {
  auto nvlsConnection = std::make_shared<NvlsConnection>();

  mscclpp::EndpointConfig cfg;
  cfg.transport = mscclpp::Transport::CudaIpc;
  cfg.device = mscclpp::DeviceType::GPU;
  cfg.nvls.numDevices = allRanks.size();
  cfg.nvls.bufferSize = bufferSize;
  if (comm->bootstrap()->getRank() == allRanks[0]) {
    cfg.nvls.isRoot = true;
    auto rootEndpoint = comm->context()->createEndpoint(cfg);
    for (int peer = 1; peer < static_cast<int>(allRanks.size()); ++peer) {
      nvlsConnection->rootPeerConnections.push_back(comm->connect(rootEndpoint, peer).get());
    }
    cfg.nvls.isRoot = false;
    auto endpoint = comm->context()->createEndpoint(cfg);
    nvlsConnection->rootSelfConnection = comm->context()->connect(rootEndpoint, endpoint);
    nvlsConnection->connection = comm->context()->connect(endpoint, rootEndpoint);
  } else {
    cfg.nvls.isRoot = false;
    auto endpoint = comm->context()->createEndpoint(cfg);
    nvlsConnection->connection = comm->connect(endpoint, 0).get();
  }
  return nvlsConnection;
}

}  // namespace mscclpp
