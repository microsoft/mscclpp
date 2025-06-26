// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/semaphore.hpp>

#include "api.h"
#include "atomic.hpp"
#include "context.hpp"
#include "debug.h"
#include "flag.hpp"

namespace mscclpp {

struct Semaphore::Impl {
  Impl(const Flag& localFlag, const RegisteredMemory& remoteFlagMemory)
      : localFlag_(localFlag), remoteFlagMemory_(remoteFlagMemory) {}

  Flag localFlag_;
  RegisteredMemory remoteFlagMemory_;
};

Semaphore::Semaphore(const Flag& localFlag, const Flag& remoteFlag)
    : pimpl_(std::make_unique<Impl>(localFlag, remoteFlag.memory())) {}

MSCCLPP_API_CPP std::shared_ptr<Connection> Semaphore::connection() const {
  return pimpl_->localFlag_.pimpl_->connection_;
}

MSCCLPP_API_CPP const RegisteredMemory& Semaphore::localMemory() const { return pimpl_->localFlag_.memory(); }

MSCCLPP_API_CPP const RegisteredMemory& Semaphore::remoteMemory() const { return pimpl_->remoteFlagMemory_; }

static Semaphore buildSemaphoreFromConnection(Communicator& communicator, std::shared_ptr<Connection> connection) {
  auto semaphoreFuture =
      communicator.buildSemaphore(connection, communicator.remoteRankOf(*connection), communicator.tagOf(*connection));
  return semaphoreFuture.get();
}

MSCCLPP_API_CPP Host2DeviceSemaphore::Host2DeviceSemaphore(const Semaphore& semaphore)
    : semaphore_(semaphore),
      expectedInboundFlagId_(detail::gpuCallocUnique<uint64_t>()),
      outboundFlagId_(detail::gpuCallocHostUnique<uint64_t>()) {
  if (connection()->localDevice().type != DeviceType::GPU) {
    throw Error("Local endpoint device type of Host2DeviceSemaphore should be GPU", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP Host2DeviceSemaphore::Host2DeviceSemaphore(Communicator& communicator,
                                                           std::shared_ptr<Connection> connection)
    : Host2DeviceSemaphore(buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP std::shared_ptr<Connection> Host2DeviceSemaphore::connection() const { return semaphore_.connection(); }

MSCCLPP_API_CPP void Host2DeviceSemaphore::signal() {
  connection()->updateAndSync(semaphore_.remoteMemory(), 0, outboundFlagId_.get(), *outboundFlagId_ + 1);
}

MSCCLPP_API_CPP Host2DeviceSemaphore::DeviceHandle Host2DeviceSemaphore::deviceHandle() const {
  Host2DeviceSemaphore::DeviceHandle device;
  device.inboundSemaphoreId = reinterpret_cast<uint64_t*>(semaphore_.localMemory().data());
  device.expectedInboundSemaphoreId = expectedInboundFlagId_.get();
  return device;
}

MSCCLPP_API_CPP Host2HostSemaphore::Host2HostSemaphore(const Semaphore& semaphore)
    : semaphore_(semaphore),
      expectedInboundFlagId_(std::make_unique<uint64_t>()),
      outboundFlagId_(std::make_unique<uint64_t>()) {
  if (connection()->transport() == Transport::CudaIpc) {
    throw Error("Host2HostSemaphore cannot be used with CudaIpc transport", ErrorCode::InvalidUsage);
  }
  if (connection()->localDevice().type != DeviceType::CPU) {
    throw Error("Local endpoint device type of Host2HostSemaphore should be CPU", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP Host2HostSemaphore::Host2HostSemaphore(Communicator& communicator,
                                                       std::shared_ptr<Connection> connection)
    : Host2HostSemaphore(buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP std::shared_ptr<Connection> Host2HostSemaphore::connection() const { return semaphore_.connection(); }

MSCCLPP_API_CPP void Host2HostSemaphore::signal() {
  connection()->updateAndSync(semaphore_.remoteMemory(), 0, outboundFlagId_.get(), *outboundFlagId_ + 1);
}

MSCCLPP_API_CPP bool Host2HostSemaphore::poll() {
  bool signaled = (atomicLoad(reinterpret_cast<uint64_t*>(semaphore_.localMemory().data()), memoryOrderAcquire) >
                   (*expectedInboundFlagId_));
  if (signaled) (*expectedInboundFlagId_) += 1;
  return signaled;
}

MSCCLPP_API_CPP void Host2HostSemaphore::wait(int64_t maxSpinCount) {
  (*expectedInboundFlagId_) += 1;
  int64_t spinCount = 0;
  while (atomicLoad(reinterpret_cast<uint64_t*>(semaphore_.localMemory().data()), memoryOrderAcquire) <
         (*expectedInboundFlagId_)) {
    if (maxSpinCount >= 0 && spinCount++ == maxSpinCount) {
      throw Error("Host2HostSemaphore::wait timed out", ErrorCode::Timeout);
    }
  }
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::MemoryDevice2DeviceSemaphore(const Semaphore& semaphore)
    : semaphore_(semaphore),
      expectedInboundFlagId_(detail::gpuCallocUnique<uint64_t>()),
      outboundFlagId_(detail::gpuCallocUnique<uint64_t>()) {
  if (connection()->localDevice().type != DeviceType::GPU) {
    throw Error("Local endpoint device type of MemoryDevice2DeviceSemaphore should be GPU", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::MemoryDevice2DeviceSemaphore(Communicator& communicator,
                                                                           std::shared_ptr<Connection> connection)
    : MemoryDevice2DeviceSemaphore(buildSemaphoreFromConnection(communicator, connection)) {}

MSCCLPP_API_CPP std::shared_ptr<Connection> MemoryDevice2DeviceSemaphore::connection() const {
  return semaphore_.connection();
}

MSCCLPP_API_CPP MemoryDevice2DeviceSemaphore::DeviceHandle MemoryDevice2DeviceSemaphore::deviceHandle() const {
  MemoryDevice2DeviceSemaphore::DeviceHandle device;
  device.remoteInboundSemaphoreId = reinterpret_cast<uint64_t*>(semaphore_.remoteMemory().data());
  device.inboundSemaphoreId = reinterpret_cast<uint64_t*>(semaphore_.localMemory().data());
  device.expectedInboundSemaphoreId = expectedInboundFlagId_.get();
  device.outboundSemaphoreId = outboundFlagId_.get();
  return device;
};

}  // namespace mscclpp
