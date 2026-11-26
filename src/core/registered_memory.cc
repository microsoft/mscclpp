// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "registered_memory.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "context.hpp"
#include "logger.hpp"
#include "serialization.hpp"
#include "unix_socket.hpp"
#include "utils_internal.hpp"

#define MSCCLPP_CULOG_WARN(cmd)                                         \
  do {                                                                  \
    CUresult err = cmd;                                                 \
    if (err != CUDA_SUCCESS) {                                          \
      const char* errStr;                                               \
      if (cuGetErrorString(err, &errStr) != CUDA_SUCCESS) {             \
        errStr = "failed to get error string";                          \
      }                                                                 \
      WARN(mscclpp::GPU, "Call to " #cmd " failed, error is ", errStr); \
    }                                                                   \
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
    CudaDeviceGuard deviceGuard(detail::gpuIdFromAddress(data));

    localGpuIpcMemHandle = GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(data));
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    transportInfo.gpuIpcMemHandle = *localGpuIpcMemHandle;
    this->transportRecords.push_back(std::move(transportInfo));
  }
  const auto rdmaTransports = transports & getRdmaTransports();
  for (size_t i = 0; i < static_cast<size_t>(Transport::NumTransports); ++i) {
    auto transport = static_cast<Transport>(i);
    if (!rdmaTransports.has(transport)) continue;
    TransportInfo transportInfo;
    transportInfo.transport = transport;
    transportInfo.rdmaMemory = contextImpl.getRdmaProvider(transport)->registerMemory(data, size);
    this->transportRecords.push_back(std::move(transportInfo));
    INFO(NET, "RDMA mr for address ", data, " with size ", size, " is registered for ", transport);
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
  if (pimpl_->transportRecords.size() > static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw Error("Too many transport info entries", ErrorCode::InternalError);
  }
  int8_t transportCount = pimpl_->transportRecords.size();
  detail::serialize(result, transportCount);
  for (const auto& entry : pimpl_->transportRecords) {
    detail::serialize(result, entry.transport);
    if (entry.transport == Transport::CudaIpc) {
      if (entry.gpuIpcMemHandle.typeFlags == GpuIpcMemHandle::Type::None) {
        THROW(GPU, Error, ErrorCode::InternalError, "GpuIpcMemHandle type is None");
      }
      detail::serialize(result, entry.gpuIpcMemHandle);
    } else if (getRdmaTransports().has(entry.transport)) {
      entry.rdmaMemory->serialize(result);
    } else {
      THROW(GPU, Error, ErrorCode::InternalError, "Unknown transport");
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
    } else if (getRdmaTransports().has(transportInfo.transport)) {
      transportInfo.rdmaMemory = deserializeRdmaMemory(transportInfo.transport, it);
    } else {
      throw Error("Unknown transport", ErrorCode::InternalError);
    }
    this->transportRecords.push_back(std::move(transportInfo));
  }
  if (it != end) {
    throw Error("Serialization failed", ErrorCode::InternalError);
  }

  // Next decide how to set this->data
  this->data = nullptr;
  if (getHostHash() == this->hostHash && getPidHash() == this->pidHash) {
    // The memory is local to the process, so originalDataPtr is valid as is
    this->data = this->originalDataPtr;
    if (transports.has(Transport::CudaIpc)) {
      auto entry = getTransportInfo(Transport::CudaIpc);
      if ((entry.gpuIpcMemHandle.typeFlags & GpuIpcMemHandle::Type::RuntimeIpc) == 0) {
        // Query which device owns this memory
        int gpuId = detail::gpuIdFromAddress(this->data);
        int currentDevice = -1;
        MSCCLPP_CUDATHROW(cudaGetDevice(&currentDevice));

        // Only set access if we're on a different device than where memory was allocated
        if (gpuId != currentDevice) {
          detail::setReadWriteMemoryAccess(this->data, entry.gpuIpcMemHandle.baseSize);
        }
      }
    }
  } else if (transports.has(Transport::CudaIpc)) {
    // Try CudaIpc first and fall back to RDMA on failure.
    auto entry = getTransportInfo(Transport::CudaIpc);
    bool hasRdma = (transports & getRdmaTransports()).any();
    try {
      auto gpuIpcMem = GpuIpcMem::create(entry.gpuIpcMemHandle);
      this->remoteMemMap = gpuIpcMem->map();
      this->data = this->remoteMemMap.get();
    } catch (const BaseError& e) {
      if (!hasRdma) {
        throw;
      }
      bool isSameHost = (getHostHash() == this->hostHash);
      if (isSameHost) {
        WARN(GPU, "CudaIpc import failed on same host, falling back to RDMA transport: ", e.what());
      } else {
        INFO(GPU, "CudaIpc import failed on remote host, falling back to RDMA transport: ", e.what());
      }
    }
  }
  if (this->data != nullptr) {
    INFO(GPU, "Opened CUDA IPC handle at pointer ", this->data);
  }
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization)
    : Impl(serialization.begin(), serialization.end()) {}

const TransportInfo& RegisteredMemory::Impl::getTransportInfo(Transport transport) const {
  for (const auto& record : transportRecords) {
    if (record.transport == transport) {
      return record;
    }
  }
  throw Error("Transport data not found", ErrorCode::InternalError);
}

}  // namespace mscclpp
