// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "flag.hpp"

#include <algorithm>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "registered_memory.hpp"
#include "serialize.hpp"

namespace mscclpp {

static std::shared_ptr<uint64_t> gpuCallocFlagId() {
#if defined(MSCCLPP_DEVICE_HIP)
  return detail::gpuCallocUncachedShared<uint64_t>();
#else   // !defined(MSCCLPP_DEVICE_HIP)
  return detail::gpuCallocShared<uint64_t>();
#endif  // !defined(MSCCLPP_DEVICE_HIP)
}

Flag::Impl::Impl(std::shared_ptr<Connection> connection) : connection_(connection) {
  // Allocate a flag ID on the local device
  const Device& localDevice = connection_->localDevice();
  if (localDevice.type == DeviceType::CPU) {
    id_ = std::make_shared<uint64_t>(0);
  } else if (localDevice.type == DeviceType::GPU) {
    if (localDevice.id < 0) {
      throw Error("Local GPU ID is not provided", ErrorCode::InvalidUsage);
    }
    MSCCLPP_CUDATHROW(cudaSetDevice(localDevice.id));
    id_ = gpuCallocFlagId();
  } else {
    throw Error("Unsupported local device type", ErrorCode::InvalidUsage);
  }
  idMemory_ = std::move(connection->context()->registerMemory(id_.get(), sizeof(uint64_t), connection_->transport()));
}

Flag::Impl::Impl(const RegisteredMemory& idMemory, const Device& device) : idMemory_(idMemory), device_(device) {}

Flag::Flag(std::shared_ptr<Impl> pimpl) : pimpl_(std::move(pimpl)) {}

MSCCLPP_API_CPP Flag::Flag(std::shared_ptr<Connection> connection) : pimpl_(std::make_shared<Impl>(connection)) {}

MSCCLPP_API_CPP std::vector<char> Flag::serialize() const {
  auto data = pimpl_->idMemory_.serialize();
  detail::serialize(data, pimpl_->device_);
  return data;
}

MSCCLPP_API_CPP Flag Flag::deserialize(const std::vector<char>& data) {
  Device device;
  auto memEnd = data.end() - sizeof(device);
  RegisteredMemory idMemory(std::make_shared<RegisteredMemory::Impl>(data.begin(), memEnd));
  auto it = detail::deserialize(memEnd, device);
  if (it != data.end()) {
    throw Error("Flag deserialize failed", ErrorCode::InvalidUsage);
  }
  return Flag(std::make_shared<Impl>(std::move(idMemory), device));
}

MSCCLPP_API_CPP const RegisteredMemory& Flag::memory() const { return pimpl_->idMemory_; }

}  // namespace mscclpp
