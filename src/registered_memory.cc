// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "registered_memory.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "context.hpp"
#include "debug.h"
#include "serialization.hpp"
#include "utils_internal.hpp"

#define MSCCLPP_CULOG_WARN(cmd)                             \
  do {                                                      \
    CUresult err = cmd;                                     \
    if (err != CUDA_SUCCESS) {                              \
      const char* errStr;                                   \
      if (cuGetErrorString(err, &errStr) != CUDA_SUCCESS) { \
        errStr = "failed to get error string";              \
      }                                                     \
      WARN("Call to " #cmd " failed, error is %s", errStr); \
    }                                                       \
  } while (false)

namespace mscclpp {

RegisteredMemory::Impl::Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl)
    : data(data),
      originalDataPtr(data),
      size(size),
      hostHash(getHostHash()),
      pidHash(getPidHash()),
      transports(transports) {
  if (transports.has(Transport::CudaIpc)) {
    localGpuIpcHandle = GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(data));
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    transportInfo.gpuIpcMemHandle = *localGpuIpcHandle;
    this->transportInfos.emplace_back(transportInfo);
  }
  if ((transports & AllIBTransports).any()) {
    auto addIb = [&](Transport ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      this->ibMrMap[ibTransport] = contextImpl.getIbContext(ibTransport)->registerMr(data, size);
      transportInfo.ibMr = this->ibMrMap[ibTransport].get();
      transportInfo.ibLocal = true;
      transportInfo.ibMrInfo = this->ibMrMap[ibTransport]->getInfo();
      this->transportInfos.push_back(transportInfo);
      INFO(MSCCLPP_NET, "IB mr for address %p with size %ld is registered", data, size);
    };
    if (transports.has(Transport::IB0)) addIb(Transport::IB0);
    if (transports.has(Transport::IB1)) addIb(Transport::IB1);
    if (transports.has(Transport::IB2)) addIb(Transport::IB2);
    if (transports.has(Transport::IB3)) addIb(Transport::IB3);
    if (transports.has(Transport::IB4)) addIb(Transport::IB4);
    if (transports.has(Transport::IB5)) addIb(Transport::IB5);
    if (transports.has(Transport::IB6)) addIb(Transport::IB6);
    if (transports.has(Transport::IB7)) addIb(Transport::IB7);
  }
}

MSCCLPP_API_CPP RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl) : pimpl_(pimpl) {}

MSCCLPP_API_CPP RegisteredMemory::~RegisteredMemory() = default;

MSCCLPP_API_CPP void* RegisteredMemory::data() const { return pimpl_->data; }

MSCCLPP_API_CPP void* RegisteredMemory::originalDataPtr() const { return pimpl_->originalDataPtr; }

MSCCLPP_API_CPP size_t RegisteredMemory::size() const { return pimpl_->size; }

MSCCLPP_API_CPP TransportFlags RegisteredMemory::transports() const { return pimpl_->transports; }

MSCCLPP_API_CPP std::vector<char> RegisteredMemory::serialize() const {
  std::vector<char> result;
  detail::serialize(result, pimpl_->originalDataPtr);
  detail::serialize(result, pimpl_->size);
  detail::serialize(result, pimpl_->hostHash);
  detail::serialize(result, pimpl_->pidHash);
  detail::serialize(result, pimpl_->transports);
  if (pimpl_->transportInfos.size() > static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw Error("Too many transport info entries", ErrorCode::InternalError);
  }
  int8_t transportCount = pimpl_->transportInfos.size();
  detail::serialize(result, transportCount);
  for (auto& entry : pimpl_->transportInfos) {
    detail::serialize(result, entry.transport);
    if (entry.transport == Transport::CudaIpc) {
      detail::serialize(result, entry.gpuIpcMemHandle);
    } else if (AllIBTransports.has(entry.transport)) {
      detail::serialize(result, entry.ibMrInfo);
    } else {
      throw Error("Unknown transport", ErrorCode::InternalError);
    }
  }
  return result;
}

MSCCLPP_API_CPP RegisteredMemory RegisteredMemory::deserialize(const std::vector<char>& data) {
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char>::const_iterator& begin,
                             const std::vector<char>::const_iterator& end) {
  auto it = begin;
  it = detail::deserialize(it, this->originalDataPtr);
  it = detail::deserialize(it, this->size);
  it = detail::deserialize(it, this->hostHash);
  it = detail::deserialize(it, this->pidHash);
  it = detail::deserialize(it, this->transports);
  int8_t transportCount;
  it = detail::deserialize(it, transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    it = detail::deserialize(it, transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      it = detail::deserialize(it, transportInfo.gpuIpcMemHandle);
    } else if (AllIBTransports.has(transportInfo.transport)) {
      it = detail::deserialize(it, transportInfo.ibMrInfo);
      transportInfo.ibLocal = false;
    } else {
      throw Error("Unknown transport", ErrorCode::InternalError);
    }
    this->transportInfos.push_back(transportInfo);
  }
  if (it != end) {
    throw Error("Serialization failed", ErrorCode::InternalError);
  }

  // Next decide how to set this->data
  this->data = nullptr;
  if (getHostHash() == this->hostHash && getPidHash() == this->pidHash) {
    // The memory is local to the process, so originalDataPtr is valid as is
    this->data = this->originalDataPtr;
  } else if (transports.has(Transport::CudaIpc)) {
    auto entry = getTransportInfo(Transport::CudaIpc);
    this->remoteGpuIpcMem = std::make_unique<GpuIpcMem>(entry.gpuIpcMemHandle);
    this->data = this->remoteGpuIpcMem->map();
    INFO(MSCCLPP_P2P, "Opened CUDA IPC handle at pointer %p", this->data);
  }
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization)
    : Impl(serialization.begin(), serialization.end()) {}

const TransportInfo& RegisteredMemory::Impl::getTransportInfo(Transport transport) const {
  for (auto& entry : transportInfos) {
    if (entry.transport == transport) {
      return entry;
    }
  }
  throw Error("Transport data not found", ErrorCode::InternalError);
}

}  // namespace mscclpp
