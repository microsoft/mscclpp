// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <list>
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
  std::shared_ptr<void> bindMemory(CUdeviceptr devicePtr, size_t devBuffSize);

 private:
  friend class NvlsConnection;

  size_t allocateRange(size_t size);
  void freeRange(size_t offset, size_t size) noexcept;

  // Store the GpuIpcMemHandle for the multicast (only on root)
  UniqueGpuIpcMemHandle localGpuIpcMemHandle_;
  // The GpuIpcMem for multicast operations (both root and non-root)
  std::shared_ptr<GpuIpcMem> gpuIpcMem_;

  size_t minMcGran_;
  bool isRoot_;
  int numDevices_;

  // Track allocated and free ranges within the multicast buffer
  std::list<std::pair<size_t, size_t>> allocatedRanges_;
  std::list<std::pair<size_t, size_t>> freeRanges_;
};

NvlsConnection::Impl::Impl(size_t bufferSize, int numDevices) : isRoot_(true), numDevices_(numDevices) {
  // Create the multicast handle using GpuIpcMemHandle
  localGpuIpcMemHandle_ = GpuIpcMemHandle::createMulticast(bufferSize, numDevices);
  if (!localGpuIpcMemHandle_ || localGpuIpcMemHandle_->typeFlags == GpuIpcMemHandle::Type::None) {
    THROW(CONN, Error, ErrorCode::SystemError, "Failed to create multicast handle");
  }

  // Create GpuIpcMem from the handle to get access to the allocation handle
  gpuIpcMem_ = GpuIpcMem::create(*localGpuIpcMemHandle_);

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

  // Initialize free ranges with the entire buffer
  freeRanges_.emplace_back(0, localGpuIpcMemHandle_->baseSize);

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
  gpuIpcMem_ = GpuIpcMem::create(handle);

  // Initialize free ranges with the entire buffer
  freeRanges_.emplace_back(0, handle.baseSize);

  INFO(CONN, "NVLS handle is imported from root with buffer size ", handle.baseSize);
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

size_t NvlsConnection::Impl::allocateRange(size_t size) {
  if (freeRanges_.empty()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "This NVLS connection mapped more than it was supposed to");
  }
  auto it = std::find_if(freeRanges_.begin(), freeRanges_.end(),
                         [size](const std::pair<size_t, size_t>& range) { return range.second >= size; });
  if (it != freeRanges_.end()) {
    size_t offset = it->first;
    size_t rangeSize = it->second;
    if (rangeSize == size) {
      freeRanges_.erase(it);
    } else {
      it->first += size;
      it->second -= size;
    }
    allocatedRanges_.emplace_back(offset, size);
    INFO(CONN, "NVLS connection allocated ", size, " bytes at offset ", offset);
    return offset;
  }
  THROW(CONN, Error, ErrorCode::InvalidUsage, "This NVLS connection cannot map the requested devBuffSize");
}

void NvlsConnection::Impl::freeRange(size_t offset, size_t size) noexcept {
  auto it = std::find_if(
      allocatedRanges_.begin(), allocatedRanges_.end(),
      [offset, size](const std::pair<size_t, size_t>& range) { return range.first == offset && range.second == size; });
  if (it == allocatedRanges_.end()) {
    WARN(CONN, "NVLS connection tried to free a range that was not allocated");
    return;
  }
  allocatedRanges_.erase(it);
  it = std::find_if(freeRanges_.begin(), freeRanges_.end(), [offset, size](const std::pair<size_t, size_t>& range) {
    return range.first + range.second >= offset;
  });
  if (it == freeRanges_.end()) {
    freeRanges_.emplace_back(offset, size);
    return;
  }
  if (it->first + it->second == offset) {
    // merge with the previous free range if possible
    it->second += size;
    // merge with the next free range if possible
    auto nextItr = std::next(it);
    if (nextItr != freeRanges_.end() && it->first + it->second == nextItr->first) {
      it->second += nextItr->second;
      freeRanges_.erase(nextItr);
    }
    return;
  } else if (it->first == offset + size) {
    // merge with the next free range if possible
    it->first -= size;
    it->second += size;
    return;
  } else {
    freeRanges_.emplace(it, offset, size);
    return;
  }
}

std::shared_ptr<void> NvlsConnection::Impl::bindMemory(CUdeviceptr devicePtr, size_t devBuffSize) {
  // Align buffer size to minimum granularity
  devBuffSize = ((devBuffSize + minMcGran_ - 1) / minMcGran_) * minMcGran_;

  // Allocate a range in the multicast buffer
  size_t offset = allocateRange(devBuffSize);

  // mapMulticast returns a shared_ptr that handles cleanup when released
  std::shared_ptr<void> mcPtr = gpuIpcMem_->mapMulticast(numDevices_, offset, devicePtr, devBuffSize);
  INFO(CONN, "NVLS connection bound memory ", (void*)devicePtr, " to ", mcPtr.get(), " at offset ", offset, ", size ",
       devBuffSize);

  // Wrap mcPtr with an additional deleter that frees the range
  return std::shared_ptr<void>(mcPtr.get(), [self = shared_from_this(), mcPtr, offset, devBuffSize](void*) {
    // mcPtr destructor will handle unmap/unbind; we just need to free the range
    self->freeRange(offset, devBuffSize);
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
  size_t allocateRange(size_t) { throw notSupportedError; }
  void freeRange(size_t, size_t) { throw notSupportedError; }
  std::shared_ptr<void> bindMemory(CUdeviceptr, size_t) { throw notSupportedError; }

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
  SwitchChannel channel((void*)devicePtr, mcPtr, size);
  // Stamp the shared barrier resource (if any) onto the channel so that
  // SwitchChannelDeviceHandle::barrier() works without separate semaphores.
  if (barrierSem_.localBarrierFlag != nullptr) {
    channel.barrier_ = barrierSem_;
  }
  return channel;
}

void NvlsConnection::attachBarrier(int nRanks) {
  // Carve the barrier flag from this connection's own multicast region: allocate the two-element
  // flag buffer, bind it to get the multicast pointer, and keep both alive for the connection's
  // lifetime. Element 0 is the shared arrival counter, element 1 is this rank's generation counter.
  auto barrierBuffer = std::make_shared<GpuBuffer<uint64_t>>(2);
  barrierChannel_ =
      std::make_shared<SwitchChannel>(bindAllocatedMemory(CUdeviceptr(barrierBuffer->data()), barrierBuffer->bytes()));
  barrierBuffer_ = barrierBuffer->memory();
  auto* localFlag = reinterpret_cast<uint64_t*>(barrierBuffer_.get());
  barrierSem_.mcBarrierFlag = reinterpret_cast<uint64_t*>(barrierChannel_->deviceHandle().mcPtr);
  barrierSem_.localBarrierFlag = localFlag;
  barrierSem_.barrierGen = localFlag + 1;  // element 1 of the barrier buffer
  barrierSem_.nRanks = nRanks;
}

SwitchChannel::DeviceHandle SwitchChannel::deviceHandle() const {
  SwitchChannel::DeviceHandle device;
  device.devicePtr = devicePtr_;
  device.mcPtr = mcPtr_.get();
  device.bufferSize = bufferSize_;
  device.barrier_ = barrier_;
  return device;
};

void* SwitchChannel::getDevicePtr() { return devicePtr_; };

namespace {
#if (CUDA_NVLS_API_AVAILABLE)
// Perform the collective NVLS handshake among allRanks and return the connection: the lowest rank
// (root) creates the multicast, serializes it, and sends it to every peer over the bootstrap; peers
// receive and import it. `tag` isolates this handshake's messages from any other on the bootstrap.
std::shared_ptr<NvlsConnection> makeNvlsConnection(std::shared_ptr<Bootstrap> bootstrap,
                                                   const std::vector<int>& allRanks, int rank, int rootRank,
                                                   bool isRoot, size_t bufferSize, int tag) {
  std::shared_ptr<NvlsConnection> conn;
  if (isRoot) {
    conn = std::make_shared<NvlsConnection>(bufferSize, allRanks.size());
    auto serialized = conn->serialize();
    for (auto nvlsRank : allRanks) {
      if (nvlsRank != rank) bootstrap->send(serialized, nvlsRank, tag);
    }
  } else {
    std::vector<char> data;
    bootstrap->recv(data, rootRank, tag);
    conn = std::make_shared<NvlsConnection>(data);
  }
  return conn;
}

// Query the multicast granularity for `numDevices` without allocating, so the barrier reserve can be
// sized before the connection's multicast object is created. Mirrors the min/max handling used when
// the connection later computes its own granularity.
size_t nvlsMinimumGranularity(int numDevices) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = numDevices;
  prop.size = 0;
  size_t granFd = 0, granFabric = 0;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&granFd, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  if (cuMulticastGetGranularity(&granFabric, &prop, CU_MULTICAST_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
    granFabric = 0;
  }
  size_t gran = std::max(granFd, granFabric);
  INFO(CONN, "NVLS minimum granularity for ", numDevices, " devices: granFd ", granFd, ", granFabric ", granFabric,
       ", selected ", gran);
  return gran;
}
#endif  // (CUDA_NVLS_API_AVAILABLE)
}  // namespace

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

#if (CUDA_NVLS_API_AVAILABLE)
  // Back the device-side barrier with a flag carved from this connection's own multicast region,
  // rather than standing up a second multicast connection + bootstrap handshake. We append one
  // granularity chunk to the requested size and bind the flag first, so it lands at a deterministic
  // offset (0) on every rank while leaving the caller's full `bufferSize` available for later
  // bindAllocatedMemory() calls. Flag layout: element 0 is the shared arrival counter, element 1 is
  // this rank's generation counter. GpuBuffer zero-initializes both, the clean starting state the
  // barrier's monotonic (never-reset) scheme requires.
  size_t barrierReserve = nvlsMinimumGranularity(static_cast<int>(allRanks.size()));
  auto conn = makeNvlsConnection(bootstrap, allRanks, rank, rootRank, isRoot, bufferSize + barrierReserve, 0);

  conn->attachBarrier(static_cast<int>(allRanks.size()));
  return conn;
#else
  throw Error("NVLS is not supported on this CUDA version (< 12.3) or kernel version (< 5.6.0)",
              ErrorCode::InvalidUsage);
#endif  // (CUDA_NVLS_API_AVAILABLE)
}

}  // namespace mscclpp
