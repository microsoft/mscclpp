// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/core.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>

#include "api.h"
#include "endpoint.hpp"
#include "gpu_ipc_mem.hpp"
#include "logger.hpp"
#include "serialization.hpp"

namespace mscclpp {

#if (CUDA_NVLS_API_AVAILABLE)
class NvlsConnection::Impl : public std::enable_shared_from_this<NvlsConnection::Impl> {
 public:
  // For root
  Impl(size_t bufferSize, int numDevices);

  // For non-root
  Impl(const std::vector<char>& data);

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  std::vector<char> serialize();
  std::shared_ptr<char> bindMemory(CUdeviceptr devicePtr, size_t devBuffSize);

 private:
  friend class NvlsConnection;

  // Store the GpuIpcMemHandle for the multicast (only on root)
  UniqueGpuIpcMemHandle localGpuIpcMemHandle_;
  // The GpuIpcMem for multicast operations (both root and non-root)
  std::unique_ptr<GpuIpcMem> gpuIpcMem_;

  size_t minMcGran_;
  bool isRoot_;
  int numDevices_;
};

NvlsConnection::Impl::Impl(size_t bufferSize, int numDevices) : isRoot_(true), numDevices_(numDevices) {
  // Create the multicast handle using GpuIpcMemHandle
  localGpuIpcMemHandle_ = GpuIpcMemHandle::createMulticast(bufferSize, numDevices);
  if (!localGpuIpcMemHandle_ || localGpuIpcMemHandle_->typeFlags == GpuIpcMemHandle::Type::None) {
    THROW(CONN, Error, ErrorCode::SystemError, "Failed to create multicast handle");
  }

  // Create GpuIpcMem from the handle to get access to the allocation handle
  gpuIpcMem_ = std::make_unique<GpuIpcMem>(*localGpuIpcMemHandle_);

  // Compute minimum granularity for user buffer alignment
  CUmulticastObjectProp mcProp = {};
  mcProp.size = localGpuIpcMemHandle_->baseSize;
  mcProp.numDevices = numDevices_;

  size_t minMcGranPosixFd;
  mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGranPosixFd, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));

  if (localGpuIpcMemHandle_->typeFlags & GpuIpcMemHandle::Type::Fabric) {
    size_t minMcGranFabric;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGranFabric, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
    minMcGran_ = std::max(minMcGranPosixFd, minMcGranFabric);
  } else {
    minMcGran_ = minMcGranPosixFd;
  }

  INFO(CONN, "NVLS handle created on root with buffer size ", localGpuIpcMemHandle_->baseSize, ", minGranularity ",
       minMcGran_);
}

NvlsConnection::Impl::Impl(const std::vector<char>& data) : isRoot_(false) {
  auto it = data.begin();
  GpuIpcMemHandle handle;
  it = detail::deserialize(it, handle);
  it = detail::deserialize(it, minMcGran_);
  it = detail::deserialize(it, numDevices_);

  // Create GpuIpcMem from the handle to import the multicast
  gpuIpcMem_ = std::make_unique<GpuIpcMem>(handle);

  INFO(CONN, "NVLS handle is imported from root");
}

std::vector<char> NvlsConnection::Impl::serialize() {
  if (!isRoot_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "Only root NVLS connection can serialize the handle");
  }
  std::vector<char> result;
  detail::serialize(result, *localGpuIpcMemHandle_);
  detail::serialize(result, minMcGran_);
  detail::serialize(result, numDevices_);
  return result;
}

std::shared_ptr<char> NvlsConnection::Impl::bindMemory(CUdeviceptr devicePtr, size_t devBuffSize) {
  // Align buffer size to minimum granularity
  devBuffSize = ((devBuffSize + minMcGran_ - 1) / minMcGran_) * minMcGran_;

  // Use mapMulticast which handles:
  // - Adding device to multicast
  // - Binding the buffer to multicast (cuMulticastBindAddr blocks until all devices call cuMulticastAddDevice)
  // - Creating and mapping the multicast pointer
  void* mcPtr = gpuIpcMem_->mapMulticast(numDevices_, devicePtr, devBuffSize);
  INFO(CONN, "NVLS connection bound memory ", (void*)devicePtr, " to ", mcPtr, ", size ", devBuffSize);

  // Return shared_ptr that keeps Impl alive
  return std::shared_ptr<char>(static_cast<char*>(mcPtr), [self = shared_from_this()](char*) {
    // No-op deleter - cleanup happens in ~GpuIpcMem via ~Impl
  });
}

#else   // !(CUDA_NVLS_API_AVAILABLE)
class NvlsConnection::Impl {
 public:
  // use this only for the root of the NVLS
  Impl(size_t, int) { throw notSupportedError; }
  Impl(const std::vector<char>&) { throw notSupportedError; }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  std::vector<char> serialize() { throw notSupportedError; }
  size_t allocateBuffer(size_t) { throw notSupportedError; }
  void freeBuffer(size_t, size_t) { throw notSupportedError; }
  std::shared_ptr<char> bindMemory(CUdeviceptr, size_t) { throw notSupportedError; }

 private:
  Error notSupportedError =
      Error("NVLS is not supported on this CUDA version (< 12.3) or kernel version (< 5.6.0)", ErrorCode::InvalidUsage);
};
#endif  // !(CUDA_NVLS_API_AVAILABLE)

NvlsConnection::NvlsConnection(size_t bufferSize, int numDevices)
    : pimpl_(std::make_shared<Impl>(bufferSize, numDevices)) {}

NvlsConnection::NvlsConnection(const std::vector<char>& data) : pimpl_(std::make_shared<Impl>(data)) {}

std::vector<char> NvlsConnection::serialize() { return pimpl_->serialize(); }

SwitchChannel NvlsConnection::bindAllocatedMemory(CUdeviceptr devicePtr, size_t size) {
  auto mcPtr = pimpl_->bindMemory(devicePtr, size);
  return SwitchChannel((void*)devicePtr, mcPtr, size);
}

SwitchChannel::DeviceHandle SwitchChannel::deviceHandle() const {
  SwitchChannel::DeviceHandle device;
  device.devicePtr = devicePtr_;
  device.mcPtr = mcPtr_.get();
  device.bufferSize = bufferSize_;
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

  return conn;
}

}  // namespace mscclpp
