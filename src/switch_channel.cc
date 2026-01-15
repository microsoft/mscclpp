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
#include "unix_socket.hpp"

namespace mscclpp {

#if (CUDA_NVLS_API_AVAILABLE)
class NvlsConnection::Impl : public std::enable_shared_from_this<NvlsConnection::Impl> {
 public:
  // use this only for the root of the NVLS
  Impl(size_t bufferSize, int numDevices);
  Impl(const std::vector<char>& data);
  ~Impl();

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  size_t getMinMcGran() { return minMcGran_; }
  std::vector<char> serialize();
  void addDevice(int cudaDeviceId);
  size_t allocateBuffer(size_t size);
  void freeBuffer(size_t offset, size_t size) noexcept;
  std::shared_ptr<char> bindMemory(CUdeviceptr devicePtr, size_t devBuffSize);

 private:
  friend class NvlsConnection;

  CUmemGenericAllocationHandle mcHandle_;
  CUmulticastObjectProp mcProp_;
  size_t bufferSize_;
  size_t minMcGran_;
  size_t mcGran_;
  // These are only defined for multicast (NVLS) capability
  int rootFd_;
  int rootPid_;
  int mcFileDesc_;

  UnixSocketClient& socketClient_ = UnixSocketClient::instance();

  std::list<std::pair<size_t, size_t>> allocatedRanges_;
  std::list<std::pair<size_t, size_t>> freeRanges_;
};

NvlsConnection::Impl::Impl(size_t bufferSize, int numDevices) : rootFd_(-1), mcFileDesc_(-1) {
  minMcGran_ = 0;
  mcGran_ = 0;
  mcProp_ = {};
  mcProp_.size = bufferSize;
  mcProp_.numDevices = numDevices;
  mcProp_.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&mcGran_, &mcProp_, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  mcProp_.size = ((mcProp_.size + mcGran_ - 1) / mcGran_) * mcGran_;
  bufferSize_ = mcProp_.size;
  INFO(MSCCLPP_COLL, "NVLS multicast properties: size=%ld, numDevices=%d, handleTypes=%lld", mcProp_.size,
       mcProp_.numDevices, mcProp_.handleTypes);
  MSCCLPP_CUTHROW(cuMulticastCreate(&mcHandle_, &mcProp_));
  MSCCLPP_CUTHROW(
      cuMemExportToShareableHandle(&mcFileDesc_, mcHandle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
  freeRanges_.emplace_back(0, bufferSize_);
  rootPid_ = getpid();
  rootFd_ = UnixSocketServer::instance().registerFd(mcFileDesc_);

  INFO(MSCCLPP_COLL,
       "NVLS handle created on root with size %ld. minGranularity %ld and recommendedGranularity %ld buffer size is "
       "%ld, adjusted size is %ld",
       mcProp_.size, minMcGran_, mcGran_, bufferSize, bufferSize_);
}

NvlsConnection::Impl::Impl(const std::vector<char>& data) : rootFd_(-1), mcFileDesc_(-1) {
  auto it = data.begin();
  std::copy_n(it, sizeof(this->mcHandle_), reinterpret_cast<char*>(&this->mcHandle_));
  it += sizeof(this->mcHandle_);
  std::copy_n(it, sizeof(this->bufferSize_), reinterpret_cast<char*>(&this->bufferSize_));
  it += sizeof(this->bufferSize_);
  std::copy_n(it, sizeof(this->minMcGran_), reinterpret_cast<char*>(&this->minMcGran_));
  it += sizeof(this->minMcGran_);
  std::copy_n(it, sizeof(this->mcGran_), reinterpret_cast<char*>(&this->mcGran_));
  it += sizeof(this->mcGran_);
  std::copy_n(it, sizeof(this->rootPid_), reinterpret_cast<char*>(&this->rootPid_));
  it += sizeof(this->rootPid_);
  std::copy_n(it, sizeof(this->rootFd_), reinterpret_cast<char*>(&this->rootFd_));

  freeRanges_.emplace_back(0, bufferSize_);
  int mcRootFileDescFd = socketClient_.requestFd(UnixSocketServer::generateSocketPath(this->rootPid_), rootFd_);
  MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&mcHandle_, reinterpret_cast<void*>(mcRootFileDescFd),
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  ::close(mcRootFileDescFd);

  INFO(MSCCLPP_COLL, "NVLS handle was imported from root");
}

NvlsConnection::Impl::~Impl() {
  // we don't need to free multicast handle object according to NCCL.
  if (mcFileDesc_ >= 0) {
    UnixSocketServer::instance().unregisterFd(rootFd_);
    ::close(mcFileDesc_);
  }
}

std::vector<char> NvlsConnection::Impl::serialize() {
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char*>(&mcHandle_), sizeof(mcHandle_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&bufferSize_), sizeof(bufferSize_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&minMcGran_), sizeof(minMcGran_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&mcGran_), sizeof(mcGran_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&rootPid_), sizeof(rootPid_), std::back_inserter(result));
  std::copy_n(reinterpret_cast<char*>(&rootFd_), sizeof(rootFd_), std::back_inserter(result));
  return result;
}

void NvlsConnection::Impl::addDevice(int cudaDeviceId) {
  MSCCLPP_CUTHROW(cuMulticastAddDevice(mcHandle_, cudaDeviceId));
  INFO(MSCCLPP_COLL, "NVLS connection created");
}

// TODO(binyli): For cuMemMap, we can not map handle to va with offset not equal to 0.
// Then we don't need to maintain the freeRanges_ list. For different memory, we could map to different mc handle.
size_t NvlsConnection::Impl::allocateBuffer(size_t size) {
  if (freeRanges_.empty()) {
    throw Error("This NVLS connection mapped more than it was supposed to", ErrorCode::InvalidUsage);
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
    INFO(MSCCLPP_COLL, "NVLS connection allocated %ld bytes at offset %ld", size, offset);
    return offset;
  }
  throw Error("This NVLS connection cannot map the requested devBuffSize", ErrorCode::InvalidUsage);
}

void NvlsConnection::Impl::freeBuffer(size_t offset, size_t size) noexcept {
  auto it = std::find_if(
      allocatedRanges_.begin(), allocatedRanges_.end(),
      [offset, size](const std::pair<size_t, size_t>& range) { return range.first == offset && range.second == size; });
  if (it == allocatedRanges_.end()) {
    WARN("NVLS connection tried to free a buffer that was not allocated");
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

std::shared_ptr<char> NvlsConnection::Impl::bindMemory(CUdeviceptr devicePtr, size_t devBuffSize) {
  if (!isCuMemMapAllocated((void*)devicePtr)) {
    throw Error("This NVLS connection tried to bind a buffer that was not allocated with cuMemMap",
                ErrorCode::InvalidUsage);
  }

  if ((uintptr_t)devicePtr % minMcGran_ != 0) {
    WARN("NVLS connection tried to bind a buffer that is not aligned to the minimum granularity");
    throw Error("This NVLS connection tried to bind a buffer that is not aligned to the minimum granularity",
                ErrorCode::InvalidUsage);
  }
  devBuffSize = ((devBuffSize + minMcGran_ - 1) / minMcGran_) * minMcGran_;
  size_t offset = allocateBuffer(devBuffSize);
  MSCCLPP_CUTHROW(cuMulticastBindAddr(mcHandle_, offset /*mcOffset*/, devicePtr, devBuffSize, 0));

  char* mcPtr;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)(&mcPtr), devBuffSize, minMcGran_, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)(mcPtr), devBuffSize, 0, mcHandle_, 0));
  detail::setReadWriteMemoryAccess(mcPtr, devBuffSize);
  INFO(MSCCLPP_COLL, "NVLS connection bound memory %p to %p at offset %ld, size %ld", (void*)devicePtr, mcPtr, offset,
       devBuffSize);

  auto deleter = [=, self = shared_from_this()](char* ptr) {
    int deviceId;
    CUdevice device;
    MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
    MSCCLPP_CUTHROW(cuDeviceGet(&device, deviceId));
    MSCCLPP_CUTHROW(cuMemUnmap((CUdeviceptr)ptr, devBuffSize));
    MSCCLPP_CUTHROW(cuMemAddressFree((CUdeviceptr)ptr, devBuffSize));
    // Refer to NCCL, Unbind can trigger RM error if buffer is freed already by users.
    // Ignore error here, unbind will succeed anyway.
    cuMulticastUnbind(mcHandle_, device, offset, devBuffSize);
    self->freeBuffer(offset, devBuffSize);
  };

  return std::shared_ptr<char>(mcPtr, deleter);
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
  void addDevice(int) { throw notSupportedError; }
  size_t getMinMcGran() { throw notSupportedError; }

 private:
  Error notSupportedError =
      Error("NVLS is not supported on this CUDA version (< 12.3) or kernel version (< 5.6.0)", ErrorCode::InvalidUsage);
};
#endif  // !(CUDA_NVLS_API_AVAILABLE)

NvlsConnection::NvlsConnection(size_t bufferSize, int numDevices)
    : pimpl_(std::make_shared<Impl>(bufferSize, numDevices)) {}

void NvlsConnection::addDevice() {
  int cudaDeviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDeviceId));
  this->addDevice(cudaDeviceId);
}

void NvlsConnection::addDevice(int cudaDeviceId) { pimpl_->addDevice(cudaDeviceId); }

NvlsConnection::NvlsConnection(const std::vector<char>& data) : pimpl_(std::make_shared<Impl>(data)) {}

std::vector<char> NvlsConnection::serialize() { return pimpl_->serialize(); }

SwitchChannel NvlsConnection::bindAllocatedMemory(CUdeviceptr devicePtr, size_t size) {
  auto mcPtr = pimpl_->bindMemory(devicePtr, size);
  return SwitchChannel((void*)devicePtr, mcPtr, size);
}

SwitchChannel::DeviceHandle SwitchChannel::deviceHandle() const {
  SwitchChannel::DeviceHandle device;
  device.devicePtr = this->devicePtr_;
  device.mcPtr = this->mcPtr_.get();
  device.bufferSize = this->bufferSize_;
  return device;
};

void* SwitchChannel::getDevicePtr() { return devicePtr_; };

size_t NvlsConnection::getMultiCastMinGranularity() { return pimpl_->getMinMcGran(); }

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

  // Now let's synchronize all ranks
  bootstrap->groupBarrier(allRanks);
  // now it is safe to add my device
  conn->addDevice();

  // sync here to make sure all ranks have added their devices
  bootstrap->groupBarrier(allRanks);
  return conn;
}

}  // namespace mscclpp
