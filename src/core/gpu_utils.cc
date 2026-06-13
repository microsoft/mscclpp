// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstring>
#include <iterator>
#include <map>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "gpu_utils_internal.hpp"

namespace mscclpp {

AvoidCudaGraphCaptureGuard::AvoidCudaGraphCaptureGuard() : mode_(cudaStreamCaptureModeRelaxed), active_(true) {
  cudaError_t res = cudaThreadExchangeStreamCaptureMode(&mode_);
  if (isCudaTeardownError(res)) {
    // Runtime is going away; just mark inactive so destructor skips restoring.
    active_ = false;
    (void)cudaGetLastError();
  } else {
    MSCCLPP_CUDATHROW(res);
  }
}

AvoidCudaGraphCaptureGuard::~AvoidCudaGraphCaptureGuard() {
  if (!active_) return;
  (void)cudaThreadExchangeStreamCaptureMode(&mode_);
}

CudaDeviceGuard::CudaDeviceGuard(int deviceId) : deviceId_(deviceId), origDeviceId_(-1) {
  if (deviceId_ >= 0) {
    MSCCLPP_CUDATHROW(cudaGetDevice(&origDeviceId_));
    if (origDeviceId_ != deviceId_) {
      MSCCLPP_CUDATHROW(cudaSetDevice(deviceId_));
    }
  }
}

CudaDeviceGuard::~CudaDeviceGuard() {
  if (deviceId_ >= 0 && origDeviceId_ >= 0 && origDeviceId_ != deviceId_) {
    (void)cudaSetDevice(origDeviceId_);
  }
}

CudaStreamWithFlags::CudaStreamWithFlags() : stream_(nullptr) { MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_)); }

CudaStreamWithFlags::CudaStreamWithFlags(unsigned int flags) {
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_));
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStreamWithFlags::~CudaStreamWithFlags() {
  if (!empty()) (void)cudaStreamDestroy(stream_);
}

void CudaStreamWithFlags::set(unsigned int flags) {
  if (!empty()) throw Error("CudaStreamWithFlags already set", ErrorCode::InvalidUsage);
  CudaDeviceGuard deviceGuard(deviceId_);
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags));
}

bool CudaStreamWithFlags::empty() const { return stream_ == nullptr; }

GpuStream::GpuStream(std::shared_ptr<GpuStreamPool> pool, std::shared_ptr<CudaStreamWithFlags> stream)
    : pool_(pool), stream_(stream) {}

GpuStream::~GpuStream() { pool_->streams_[deviceId()].push_back(stream_); }

GpuStreamPool::GpuStreamPool() {}

GpuStream GpuStreamPool::getStream() {
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  auto& streamVec = streams_[deviceId];
  if (!streamVec.empty()) {
    auto stream = streamVec.back();
    streamVec.pop_back();
    return GpuStream(gpuStreamPool(), stream);
  }
  return GpuStream(gpuStreamPool(), std::make_shared<CudaStreamWithFlags>(cudaStreamNonBlocking));
}

void GpuStreamPool::clear() { streams_.clear(); }

// A global pool instance
static std::shared_ptr<GpuStreamPool> gGpuStreamPool_;

std::shared_ptr<GpuStreamPool> gpuStreamPool() {
  if (!gGpuStreamPool_) {
    gGpuStreamPool_ = std::make_shared<GpuStreamPool>();
  }
  return gGpuStreamPool_;
}

namespace detail {

int gpuIdFromAddress(void* ptr) {
  int deviceId;
  auto res = cuPointerGetAttribute(&deviceId, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, reinterpret_cast<CUdeviceptr>(ptr));
  if (res == CUDA_ERROR_INVALID_VALUE) {
    // not a GPU address
    return -1;
  } else {
    MSCCLPP_CUTHROW(res);
  }
  return deviceId;
}

/// set memory access permission to read-write
/// @param base Base memory pointer.
/// @param size Size of the memory.
void setReadWriteMemoryAccess(void* base, size_t size) {
  CUmemAccessDesc accessDesc = {};
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess((CUdeviceptr)base, size, &accessDesc, 1));
}

void* gpuCalloc(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

void* gpuCallocHost(size_t bytes, unsigned int flags) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

#if defined(MSCCLPP_USE_ROCM)
void* gpuCallocUncached(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(hipExtMallocWithFlags((void**)&ptr, bytes, hipDeviceMallocUncached));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}
#endif  // defined(MSCCLPP_USE_ROCM)

#if (CUDA_NVLS_API_AVAILABLE)
size_t getCuAllocationGranularity(CUmemAllocationGranularity_flags granFlag) {
  size_t gran = 0;
  int deviceId = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = deviceId;
  prop.requestedHandleTypes =
      (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  cuMemGetAllocationGranularity(&gran, &prop, granFlag);
  return gran;
}

size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag) {
  size_t gran = 0;
  int numDevices = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&numDevices));

  CUmulticastObjectProp prop = {};
  prop.size = size;
  // This is a dummy value, it might affect the granularity in the future
  prop.numDevices = numDevices;
  prop.handleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  prop.flags = 0;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&gran, &prop, granFlag));
  return gran;
}

void* gpuCallocPhysical(size_t bytes, size_t gran, size_t align) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  int deviceId = -1;
  CUdevice currentDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUTHROW(cuDeviceGet(&currentDevice, deviceId));

  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int isFabricSupported;
  MSCCLPP_CUTHROW(
      cuDeviceGetAttribute(&isFabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDevice));
  if (isFabricSupported) {
    requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = (CUmemAllocationHandleType)(requestedHandleTypes);
  prop.location.id = currentDevice;

  if (gran == 0) {
    gran = getMulticastGranularity(bytes, CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }

  // allocate physical memory
  CUmemGenericAllocationHandle memHandle;
  size_t nbytes = (bytes + gran - 1) / gran * gran;
  CUresult result = cuMemCreate(&memHandle, nbytes, &prop, 0);
  if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC &&
      (result == CUDA_ERROR_NOT_PERMITTED || result == CUDA_ERROR_NOT_SUPPORTED)) {
    requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
    MSCCLPP_CUTHROW(cuMemCreate(&memHandle, nbytes, &prop, 0));
  } else {
    MSCCLPP_CUTHROW(result);
  }

  if (align == 0) {
    align = getMulticastGranularity(nbytes, CU_MULTICAST_GRANULARITY_MINIMUM);
  }

  void* devicePtr = nullptr;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)&devicePtr, nbytes, align, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)devicePtr, nbytes, 0, memHandle, 0));
  setReadWriteMemoryAccess(devicePtr, nbytes);
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(cudaMemsetAsync(devicePtr, 0, nbytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  return devicePtr;
}
#endif  // CUDA_NVLS_API_AVAILABLE

void gpuFree(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cudaFree(ptr));
}

void gpuFreeHost(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cudaFreeHost(ptr));
}

#if (CUDA_NVLS_API_AVAILABLE)
void gpuFreePhysical(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRetainAllocationHandle(&handle, ptr));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRelease(handle));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  MSCCLPP_CUTHROW_IGNORE(cuMemUnmap((CUdeviceptr)ptr, size));
  MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cuMemRelease(handle));
  MSCCLPP_CUTHROW_IGNORE(cuMemAddressFree((CUdeviceptr)ptr, size));
}
#endif  // CUDA_NVLS_API_AVAILABLE

void gpuMemcpyAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
}

void gpuMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

void gpuMemset(void* ptr, int value, size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, value, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

class GpuBufferPoolStorage : public std::enable_shared_from_this<GpuBufferPoolStorage> {
 public:
  GpuBufferPoolStorage(size_t bytes, GpuBufferGranularity granularity);
  std::shared_ptr<GpuBufferPool::Buffer> allocate(size_t bytes, size_t alignment);
  void release(size_t offset) noexcept;
  size_t bytes() const;
  size_t freeBytes() const;
  size_t activeBytes() const;
  char* data();
  int deviceId() const;

 private:
  struct Block {
    size_t offset;
    size_t bytes;
    size_t reservedOffset;
    size_t reservedBytes;
  };

  static size_t alignUp(size_t offset, size_t alignment);
  Block reserveBlock(size_t bytes, size_t alignment);
  void releaseBlock(size_t offset, size_t bytes) noexcept;

  mutable std::mutex mutex_;
  GpuBuffer<char> buffer_;
  std::map<size_t, size_t> freeBlocks_;
  std::unordered_map<size_t, Block> activeBlocks_;
};

GpuBufferPoolStorage::GpuBufferPoolStorage(size_t bytes, GpuBufferGranularity granularity)
    : buffer_(bytes, granularity) {
  if (bytes == 0) {
    throw Error("GpuBufferPool size must be positive.", ErrorCode::InvalidUsage);
  }
  freeBlocks_[0] = buffer_.bytes();
}

size_t GpuBufferPoolStorage::alignUp(size_t offset, size_t alignment) {
  if (alignment == 0) {
    throw Error("GpuBufferPool allocation alignment must be positive.", ErrorCode::InvalidUsage);
  }
  size_t remainder = offset % alignment;
  if (remainder == 0) {
    return offset;
  }
  return offset + alignment - remainder;
}

GpuBufferPoolStorage::Block GpuBufferPoolStorage::reserveBlock(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    throw Error("GpuBufferPool allocation size must be positive.", ErrorCode::InvalidUsage);
  }
  for (auto it = freeBlocks_.begin(); it != freeBlocks_.end(); ++it) {
    size_t blockOffset = it->first;
    size_t blockBytes = it->second;
    size_t alignedOffset = alignUp(blockOffset, alignment);
    size_t prefixBytes = alignedOffset - blockOffset;
    if (prefixBytes > blockBytes || bytes > blockBytes - prefixBytes) {
      continue;
    }

    size_t suffixOffset = alignedOffset + bytes;
    size_t reservedBytes = prefixBytes + bytes;
    size_t suffixBytes = blockBytes - reservedBytes;
    freeBlocks_.erase(it);
    if (suffixBytes > 0) {
      freeBlocks_[suffixOffset] = suffixBytes;
    }
    Block block{alignedOffset, bytes, blockOffset, reservedBytes};
    activeBlocks_[alignedOffset] = block;
    return block;
  }
  throw Error("GpuBufferPool does not have enough free memory for the requested allocation.", ErrorCode::InvalidUsage);
}

void GpuBufferPoolStorage::releaseBlock(size_t offset, size_t bytes) noexcept {
  auto next = freeBlocks_.lower_bound(offset);
  if (next != freeBlocks_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      offset = prev->first;
      bytes += prev->second;
      next = freeBlocks_.erase(prev);
    }
  }
  if (next != freeBlocks_.end() && offset + bytes == next->first) {
    bytes += next->second;
    freeBlocks_.erase(next);
  }
  freeBlocks_[offset] = bytes;
}

std::shared_ptr<GpuBufferPool::Buffer> GpuBufferPoolStorage::allocate(size_t bytes, size_t alignment) {
  std::lock_guard<std::mutex> lock(mutex_);
  Block block = reserveBlock(bytes, alignment);
  return std::shared_ptr<GpuBufferPool::Buffer>(new GpuBufferPool::Buffer(shared_from_this(), block.offset, bytes));
}

void GpuBufferPoolStorage::release(size_t offset) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  auto active = activeBlocks_.find(offset);
  if (active == activeBlocks_.end()) {
    return;
  }
  Block block = active->second;
  activeBlocks_.erase(active);
  releaseBlock(block.reservedOffset, block.reservedBytes);
}

size_t GpuBufferPoolStorage::bytes() const { return buffer_.bytes(); }

size_t GpuBufferPoolStorage::freeBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t freeBytes = 0;
  for (auto const& block : freeBlocks_) {
    freeBytes += block.second;
  }
  return freeBytes;
}

size_t GpuBufferPoolStorage::activeBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t activeBytes = 0;
  for (auto const& block : activeBlocks_) {
    activeBytes += block.second.bytes;
  }
  return activeBytes;
}

char* GpuBufferPoolStorage::data() { return buffer_.data(); }

int GpuBufferPoolStorage::deviceId() const { return buffer_.deviceId(); }

}  // namespace detail

GpuBufferPool::Buffer::Buffer(std::shared_ptr<detail::GpuBufferPoolStorage> storage, size_t offset, size_t bytes)
    : storage_(std::move(storage)), offset_(offset), bytes_(bytes) {}

GpuBufferPool::Buffer::~Buffer() { storage_->release(offset_); }

size_t GpuBufferPool::Buffer::bytes() const { return bytes_; }

size_t GpuBufferPool::Buffer::offset() const { return offset_; }

char* GpuBufferPool::Buffer::data() const { return storage_->data() + offset_; }

int GpuBufferPool::Buffer::deviceId() const { return storage_->deviceId(); }

GpuBufferPool::GpuBufferPool(size_t bytes, GpuBufferGranularity granularity)
    : storage_(std::make_shared<detail::GpuBufferPoolStorage>(bytes, granularity)) {}

std::shared_ptr<GpuBufferPool::Buffer> GpuBufferPool::allocate(size_t bytes, size_t alignment) {
  return storage_->allocate(bytes, alignment);
}

size_t GpuBufferPool::bytes() const { return storage_->bytes(); }

size_t GpuBufferPool::freeBytes() const { return storage_->freeBytes(); }

size_t GpuBufferPool::activeBytes() const { return storage_->activeBytes(); }

char* GpuBufferPool::data() { return storage_->data(); }

int GpuBufferPool::deviceId() const { return storage_->deviceId(); }

bool isNvlsSupported() {
  if (env()->forceDisableNvls) {
    return false;
  }
  [[maybe_unused]] static bool result = false;
  [[maybe_unused]] static bool isChecked = false;
#if (CUDA_NVLS_API_AVAILABLE)
  if (!isChecked) {
    int deviceId;
    int isMulticastSupported;
    CUdevice dev;
    MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
    MSCCLPP_CUTHROW(cuDeviceGet(&dev, deviceId));
    MSCCLPP_CUTHROW(cuDeviceGetAttribute(&isMulticastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    result = (isMulticastSupported == 1);
    isChecked = true;
    return result;
  }
  return result;
#endif
  return false;
}

bool isCuMemMapAllocated([[maybe_unused]] void* ptr) {
#if defined(MSCCLPP_USE_ROCM)
  return false;
#else
  CUmemGenericAllocationHandle handle;
  CUresult result = cuMemRetainAllocationHandle(&handle, ptr);
  if (result != CUDA_SUCCESS) {
    return false;
  }
  MSCCLPP_CUTHROW(cuMemRelease(handle));
  return true;
#endif
}

}  // namespace mscclpp
