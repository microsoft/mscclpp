// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu_ipc_mem.hpp"

#include <linux/version.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mscclpp/gpu_utils.hpp>
#include <sstream>

#include "debug.h"

namespace mscclpp {

static int dupFdFromPid([[maybe_unused]] pid_t pid, [[maybe_unused]] int targetFd) {
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0))
  // Linux pidfd based cross-process fd duplication
  int pidfd = syscall(SYS_pidfd_open, pid, 0);
  if (pidfd < 0) {
    return -1;
  }
  int dupfd = syscall(SYS_pidfd_getfd, pidfd, targetFd, 0);
  if (dupfd < 0) {
    close(pidfd);
    return -1;
  }
  close(pidfd);
  return dupfd;
#else
  return -1;
#endif
}

void GpuIpcMemHandle::deleter(GpuIpcMemHandle* handle) {
  if (handle) {
    if (handle->typeFlags & GpuIpcMemHandle::Type::PosixFd) {
      ::close(handle->posixFd.fd);
    }
  }
}

UniqueGpuIpcMemHandle GpuIpcMemHandle::create(const CUdeviceptr ptr) {
  auto handle = UniqueGpuIpcMemHandle(new GpuIpcMemHandle(), &GpuIpcMemHandle::deleter);
  handle->typeFlags = GpuIpcMemHandle::Type::None;

  CUdeviceptr basePtr;
  size_t sz;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&basePtr, &sz, ptr));
  if (sz == 0) return handle;  // No valid memory range found
  handle->baseSize = sz;
  handle->offsetFromBase = size_t(ptr) - size_t(basePtr);

  // Runtime IPC handle
  cudaIpcGetMemHandle(&handle->runtimeIpc.handle, (void*)basePtr);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    handle->typeFlags |= GpuIpcMemHandle::Type::RuntimeIpc;
  }

  CUmemGenericAllocationHandle allocHandle;
  CUresult res = cuMemRetainAllocationHandle(&allocHandle, (void*)basePtr);
  if (res == CUDA_ERROR_NOT_SUPPORTED || res == CUDA_ERROR_INVALID_VALUE) {
    // Not supported on this platform or not mapped by cuMem API
    return handle;
  }
  MSCCLPP_CUTHROW(res);

  // POSIX FD handle
  if (cuMemExportToShareableHandle(&(handle->posixFd.fd), allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
      CUDA_SUCCESS) {
    handle->posixFd.pid = getpid();
    handle->typeFlags |= GpuIpcMemHandle::Type::PosixFd;
  }

  // FABRIC handle
  if (cuMemExportToShareableHandle(&(handle->fabric.handle), allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0) ==
      CUDA_SUCCESS) {
    handle->typeFlags |= GpuIpcMemHandle::Type::Fabric;
  }

  MSCCLPP_CUTHROW(cuMemRelease(allocHandle));

  return handle;
}

UniqueGpuIpcMemHandle GpuIpcMemHandle::createMulticast(size_t bufferSize, int numDevices) {
#if (CUDA_NVLS_API_AVAILABLE)
  auto handle = UniqueGpuIpcMemHandle(new GpuIpcMemHandle(), &GpuIpcMemHandle::deleter);

  CUmulticastObjectProp prop;
  prop.size = bufferSize;
  prop.numDevices = numDevices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t minMcGran;
  size_t recMcGran;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&recMcGran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  // update bufferSize
  bufferSize = ((bufferSize + recMcGran - 1) / recMcGran) * recMcGran;

  handle->typeFlags = GpuIpcMemHandle::Type::None;
  handle->baseSize = bufferSize;
  handle->offsetFromBase = 0;

  CUmemGenericAllocationHandle allocHandle;
  prop.size = bufferSize;
  MSCCLPP_CUTHROW(cuMulticastCreate(&allocHandle, &prop));

  // POSIX FD handle
  if (cuMemExportToShareableHandle(&(handle->posixFd.fd), allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
      CUDA_SUCCESS) {
    handle->posixFd.pid = getpid();
    handle->typeFlags |= GpuIpcMemHandle::Type::PosixFd;
  }

  // FABRIC handle
  if (cuMemExportToShareableHandle(&(handle->fabric.handle), allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0) ==
      CUDA_SUCCESS) {
    handle->typeFlags |= GpuIpcMemHandle::Type::Fabric;
  }
  return handle;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  throw Error("NVLS is not supported on this CUDA version (< 12.3) or kernel version (< 5.6.0)",
              ErrorCode::InvalidUsage);
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

GpuIpcMem::GpuIpcMem(const GpuIpcMemHandle& handle) : handle_(handle), isMulticast_(false) {
  type_ = GpuIpcMemHandle::Type::None;
  basePtr_ = nullptr;
  baseSize_ = 0;
  dataPtr_ = nullptr;
  dataSize_ = 0;
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::Fabric)) {
    if (cuMemImportFromShareableHandle(&allocHandle_, (void*)handle_.fabric.handle, CU_MEM_HANDLE_TYPE_FABRIC) ==
        CUDA_SUCCESS) {
      type_ = GpuIpcMemHandle::Type::Fabric;
    }
  }
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::PosixFd)) {
    int dupfd = dupFdFromPid(handle_.posixFd.pid, handle_.posixFd.fd);
    if (dupfd != -1) {
      if (cuMemImportFromShareableHandle(&allocHandle_, (void*)(uintptr_t)dupfd,
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS) {
        type_ = GpuIpcMemHandle::Type::PosixFd;
      }
      close(dupfd);
    }
  }
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::RuntimeIpc)) {
    cudaIpcOpenMemHandle(&basePtr_, handle_.runtimeIpc.handle, cudaIpcMemLazyEnablePeerAccess);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
      baseSize_ = handle_.baseSize;
      dataPtr_ = static_cast<void*>(static_cast<char*>(basePtr_) + handle_.offsetFromBase);
      dataSize_ = handle_.baseSize - handle_.offsetFromBase;
      type_ = GpuIpcMemHandle::Type::RuntimeIpc;
      return;
    }
  }
}

GpuIpcMem::~GpuIpcMem() {
  if (type_ == GpuIpcMemHandle::Type::RuntimeIpc) {
    cudaIpcCloseMemHandle(basePtr_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      WARN("Failed to close CUDA IPC handle at pointer %p: %s", basePtr_, cudaGetErrorString(err));
    }
  } else if (type_ == GpuIpcMemHandle::Type::PosixFd || type_ == GpuIpcMemHandle::Type::Fabric) {
    CUresult res;
    const char* errStr;
    int deviceId;
    res = cuPointerGetAttribute(&deviceId, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)basePtr_);
    if (res != CUDA_SUCCESS) {
      cuGetErrorString(res, &errStr);
      WARN("Failed to get device ordinal for pointer %p: %s", basePtr_, errStr);
      deviceId = -1;
    } else if (deviceId < 0) {
      WARN("Invalid device ordinal %d for pointer %p", deviceId, basePtr_);
    }
    if (basePtr_) {
      res = cuMemUnmap((CUdeviceptr)basePtr_, baseSize_);
      if (res != CUDA_SUCCESS) {
        cuGetErrorString(res, &errStr);
        WARN("Failed to unmap CUDA memory at pointer %p: %s", basePtr_, errStr);
      }
      res = cuMemAddressFree((CUdeviceptr)basePtr_, baseSize_);
      if (res != CUDA_SUCCESS) {
        cuGetErrorString(res, &errStr);
        WARN("Failed to free CUDA memory at pointer %p: %s", basePtr_, errStr);
      }
    }
    if (isMulticast_ && deviceId >= 0) {
      CUdevice device;
      if (cuDeviceGet(&device, deviceId) == CUDA_SUCCESS) {
        cuMulticastUnbind(allocHandle_, device, 0, baseSize_);
      }
    }
    res = cuMemRelease(allocHandle_);
    if (res != CUDA_SUCCESS) {
      cuGetErrorString(res, &errStr);
      WARN("Failed to release CUDA memory allocation handle: %s", errStr);
    }
  }
}

void *GpuIpcMem::map() {
  if (type_ == GpuIpcMemHandle::Type::None) {
    throw Error("GpuIpcMemHandle type is None, cannot map memory", ErrorCode::InvalidUsage);
  } else if (dataPtr_ != nullptr) {
    // Already mapped
    return dataPtr_;
  }

  size_t pageSize = getpagesize();
  if (handle_.baseSize % pageSize) {
    std::stringstream ss;
    ss << "Tried to map remote GPU memory with size " << handle_.baseSize
       << " that is not a multiple of the local host page size " << pageSize;
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }

  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));

  CUdeviceptr base;
  size_t minGran;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = deviceId;
  MSCCLPP_CUTHROW(cuMemGetAllocationGranularity(&minGran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  MSCCLPP_CUTHROW(cuMemAddressReserve(&base, handle_.baseSize, minGran, 0, 0));
  MSCCLPP_CUTHROW(cuMemMap(base, handle_.baseSize, 0, allocHandle_, 0));

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(base, handle_.baseSize, &accessDesc, 1));

  basePtr_ = (void*)base;
  baseSize_ = handle_.baseSize;
  dataPtr_ = static_cast<void*>(static_cast<char*>(basePtr_) + handle_.offsetFromBase);
  dataSize_ = handle_.baseSize - handle_.offsetFromBase;
  return dataPtr_;
}

void *GpuIpcMem::mapMulticast(int numDevices, const CUdeviceptr devicePtr) {
#if (CUDA_NVLS_API_AVAILABLE)
  if (type_ != GpuIpcMemHandle::Type::PosixFd && type_ != GpuIpcMemHandle::Type::Fabric) {
    throw Error("GpuIpcMemHandle type is not PosixFd or Fabric, cannot map multicast memory", ErrorCode::InvalidUsage);
  }
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUTHROW(cuMulticastAddDevice(allocHandle_, deviceId));

  size_t minMcGran;
  CUmulticastObjectProp prop;
  prop.size = handle_.baseSize;
  prop.numDevices = numDevices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;

  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&minMcGran, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));

  CUdeviceptr bufferPtr;
  if (devicePtr != 0) {
    if (!isCuMemMapAllocated((void*)devicePtr)) {
      throw Error("This NVLS connection tried to bind a buffer that was not allocated with cuMemMap",
                  ErrorCode::InvalidUsage);
    }
    if ((uintptr_t)devicePtr % minMcGran != 0) {
      throw Error("This NVLS connection tried to bind a buffer that is not aligned to the minimum granularity",
                  ErrorCode::InvalidUsage);
    }
    bufferPtr = devicePtr;
  } else {
    multicastBuffer_ = GpuBuffer<uint8_t>(handle_.baseSize).memory();
    bufferPtr = (CUdeviceptr)(multicastBuffer_.get());
  }

  // will block until all devices call cuMulticastAddDevice()
  MSCCLPP_CUTHROW(cuMulticastBindAddr(allocHandle_, 0, bufferPtr, handle_.baseSize, 0));

  CUdeviceptr mcPtr;
  MSCCLPP_CUTHROW(cuMemAddressReserve(&mcPtr, handle_.baseSize, minMcGran, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap(mcPtr, handle_.baseSize, 0, allocHandle_, 0));

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(mcPtr, handle_.baseSize, &accessDesc, 1));

  basePtr_ = (void*)mcPtr;
  baseSize_ = handle_.baseSize;
  dataPtr_ = basePtr_;
  dataSize_ = baseSize_;
  isMulticast_ = true;
  return dataPtr_;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  throw Error("NVLS is not supported on this CUDA version (< 12.3) or kernel version (< 5.6.0)",
              ErrorCode::InvalidUsage);
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

void *GpuIpcMem::data() const {
  if (!dataPtr_) {
    throw Error("GpuIpcMem data pointer is null. Call map() first.", ErrorCode::InvalidUsage);
  }
  return dataPtr_;
}

}  // namespace mscclpp
