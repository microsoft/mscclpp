// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "flag.hpp"

#include <mscclpp/gpu_utils.hpp>

namespace mscclpp {

static std::shared_ptr<uint64_t> gpuCallocFlagId() {
#if defined(MSCCLPP_DEVICE_HIP)
  return detail::gpuCallocUncachedShared<uint64_t>();
#else   // !defined(MSCCLPP_DEVICE_HIP)
  return detail::gpuCallocShared<uint64_t>();
#endif  // !defined(MSCCLPP_DEVICE_HIP)
}

Flag::Impl::Impl(std::shared_ptr<Connection> connection, Device device, Context& context)
    : connection_(connection),
      device_(device),
      id_(device == Device::GPU ? gpuCallocFlagId() : std::make_shared<uint64_t>(0)),
      idMemory_(std::move(context.registerMemory(id_.get(), sizeof(uint64_t), connection_->transport()))) {}

Flag::Flag(std::shared_ptr<Impl> pimpl) : pimpl_(std::move(pimpl)) {}

const RegisteredMemory& Flag::memory() const { return pimpl_->idMemory_; }

}  // namespace mscclpp
