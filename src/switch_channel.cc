// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <mscclpp/core.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>

#include "api.h"
#include "debug.h"
#include "endpoint.hpp"
#include "gpu_ipc_mem.hpp"
#include "serialization.hpp"

namespace mscclpp {

struct NvlsConnection::Impl : public std::enable_shared_from_this<NvlsConnection::Impl> {
  // use this only for the root of the NVLS
  Impl(size_t bufferSize, int numDevices);
  Impl(const std::vector<char>& data);

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  std::vector<char> serialize();
  void bindMemory(CUdeviceptr devicePtr, size_t devBuffSize);

  int numDevices_;
  UniqueGpuIpcMemHandle gpuIpcMemHandle_;
  std::unique_ptr<GpuIpcMem> gpuIpcMem_;
};

NvlsConnection::Impl::Impl(size_t bufferSize, int numDevices)
    : numDevices_(numDevices), gpuIpcMemHandle_(GpuIpcMemHandle::createMulticast(bufferSize, numDevices)) {}

NvlsConnection::Impl::Impl(const std::vector<char>& data) : gpuIpcMemHandle_(nullptr, &GpuIpcMemHandle::deleter) {
  GpuIpcMemHandle hdl;
  detail::deserialize(data.begin(), hdl);
  gpuIpcMem_ = std::make_unique<GpuIpcMem>(hdl);
}

std::vector<char> NvlsConnection::Impl::serialize() {
  std::vector<char> result;
  detail::serialize(result, *gpuIpcMemHandle_);
  return result;
}

void NvlsConnection::Impl::bindMemory(CUdeviceptr devicePtr, size_t devBuffSize) {
  if (!gpuIpcMem_) {
    gpuIpcMem_ = std::make_unique<GpuIpcMem>(*gpuIpcMemHandle_);
  }
  gpuIpcMem_->mapMulticast(numDevices_, devicePtr);
}

NvlsConnection::NvlsConnection(size_t bufferSize, int numDevices)
    : pimpl_(std::make_shared<Impl>(bufferSize, numDevices)) {}

NvlsConnection::NvlsConnection(const std::vector<char>& data) : pimpl_(std::make_shared<Impl>(data)) {}

std::vector<char> NvlsConnection::serialize() { return pimpl_->serialize(); }

void NvlsConnection::bindMemory(CUdeviceptr devicePtr, size_t size) { pimpl_->bindMemory(devicePtr, size); }

SwitchChannel NvlsConnection::bindAllocatedMemory(CUdeviceptr devicePtr, size_t size) {
  pimpl_->bindMemory(devicePtr, size);
  return SwitchChannel(std::shared_ptr<NvlsConnection>(this));
}

void* NvlsConnection::devicePtr() const { return pimpl_->gpuIpcMem_->multicastBuffer(); }

void* NvlsConnection::mcPtr() const { return pimpl_->gpuIpcMem_->data(); }

size_t NvlsConnection::bufferSize() const { return pimpl_->gpuIpcMem_->size(); }

SwitchChannel::SwitchChannel(std::shared_ptr<NvlsConnection> conn) {
  devicePtr_ = conn->devicePtr();
  mcPtr_ = conn->mcPtr();
  bufferSize_ = conn->bufferSize();
}

SwitchChannel::DeviceHandle SwitchChannel::deviceHandle() const {
  SwitchChannel::DeviceHandle device;
  device.devicePtr = this->devicePtr_;
  device.mcPtr = this->mcPtr_;
  device.bufferSize = this->bufferSize_;
  return device;
};

void* SwitchChannel::getDevicePtr() { return devicePtr_; };

MSCCLPP_API_CPP std::shared_ptr<NvlsConnection> connectNvlsCollective(std::shared_ptr<Communicator> comm,
                                                                      std::vector<int> allRanks, size_t bufferSize) {
  auto bootstrap = comm->bootstrap();
  int rank = bootstrap->getRank();
  bool isRoot = false;
  bool amongAllRanks = false;
  int rootRank = allRanks[0];
  for (auto nvlsRank : allRanks) {
    if (nvlsRank == rank) amongAllRanks = true;
    rootRank = std::min(rootRank, nvlsRank);
  }
  if (amongAllRanks == false) {
    throw Error("rank is not among allRanks", ErrorCode::InvalidUsage);
  }
  if (rootRank == rank) isRoot = true;

  std::shared_ptr<NvlsConnection> conn;
  if (isRoot) {
    conn = std::make_shared<NvlsConnection>(bufferSize, allRanks.size());
    auto serialized = conn->serialize();
    for (auto nvlsRank : allRanks) {
      if (nvlsRank != rank) bootstrap->send(serialized, nvlsRank, 0);
    }
  } else {
    std::vector<char> data;
    bootstrap->recv(data, rootRank, 0);
    conn = std::make_shared<NvlsConnection>(data);
  }

  // sync here to make sure all ranks have added their devices
  bootstrap->groupBarrier(allRanks);
  return conn;
}

}  // namespace mscclpp
