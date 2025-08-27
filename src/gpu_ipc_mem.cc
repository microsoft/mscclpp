// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu_ipc_mem.hpp"

#include <linux/version.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mscclpp/gpu_utils.hpp>
#include <sstream>

#include "debug.h"

namespace std {

std::string to_string(const mscclpp::GpuIpcMemHandle::TypeFlags& typeFlags) {
  std::stringstream ss;
  if (typeFlags & mscclpp::GpuIpcMemHandle::Type::RuntimeIpc) {
    ss << "RuntimeIpc|";
  }
  if (typeFlags & mscclpp::GpuIpcMemHandle::Type::PosixFd) {
    ss << "PosixFd|";
  }
  if (typeFlags & mscclpp::GpuIpcMemHandle::Type::Fabric) {
    ss << "Fabric|";
  }
  std::string ret = ss.str();
  if (!ret.empty()) {
    ret.pop_back();  // Remove the trailing '|'
    return ret;
  }
  return "None";
}

}  // namespace std

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

[[maybe_unused]] static bool isFabricMemHandleAvailable() {
#if (CUDA_NVLS_API_AVAILABLE)
  CUdevice currentDevice;
  int isFabricSupported;
  MSCCLPP_CUTHROW(cuCtxGetDevice(&currentDevice));
  MSCCLPP_CUTHROW(
      cuDeviceGetAttribute(&isFabricSupported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDevice));
  if (isFabricSupported == 0) {
    return false;
  }

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = currentDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t minGran;
  MSCCLPP_CUTHROW(cuMemGetAllocationGranularity(&minGran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // try allocating minimal amount of memory
  CUmemGenericAllocationHandle memHandle;
  CUresult result = cuMemCreate(&memHandle, minGran, &prop, 0);
  if (result == CUDA_ERROR_NOT_PERMITTED || result == CUDA_ERROR_NOT_SUPPORTED) {
    // unprivileged user or old kernel version
    return false;
  } else {
    MSCCLPP_CUTHROW(result);
  }

  // it worked; cleanup now
  MSCCLPP_CUTHROW(cuMemRelease(memHandle));
  return true;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  return false;
#endif  // !(CUDA_NVLS_API_AVAILABLE)
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
  cudaError_t err = cudaIpcGetMemHandle(&handle->runtimeIpc.handle, (void*)basePtr);
  if (err == cudaSuccess) {
    handle->typeFlags |= GpuIpcMemHandle::Type::RuntimeIpc;
  } else {
    (void)cudaGetLastError();
  }

#if !defined(MSCCLPP_DEVICE_HIP)  // Remove when HIP fully supports virtual memory management APIs
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
#endif  // !defined(MSCCLPP_DEVICE_HIP)

  return handle;
}

UniqueGpuIpcMemHandle GpuIpcMemHandle::createMulticast([[maybe_unused]] size_t bufferSize,
                                                       [[maybe_unused]] int numDevices) {
#if (CUDA_NVLS_API_AVAILABLE)
  if (bufferSize == 0) {
    throw Error("Multicasting buffer size should be positive", ErrorCode::InvalidUsage);
  }
  if (numDevices < 1) {
    throw Error("Multicasting number of devices should be positive", ErrorCode::InvalidUsage);
  }
  auto handle = UniqueGpuIpcMemHandle(new GpuIpcMemHandle(), &GpuIpcMemHandle::deleter);

  bool isFabricAvailable = isFabricMemHandleAvailable();

  // get granularity
  size_t recMcGran;
  CUmulticastObjectProp prop;
  prop.size = bufferSize;
  prop.numDevices = numDevices;
  if (isFabricAvailable) {
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&recMcGran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  CUmemGenericAllocationHandle allocHandle;
  size_t baseSize = ((bufferSize + recMcGran - 1) / recMcGran) * recMcGran;
  prop.size = baseSize;
  MSCCLPP_CUTHROW(cuMulticastCreate(&allocHandle, &prop));

  handle->baseSize = baseSize;
  handle->offsetFromBase = 0;
  handle->typeFlags = GpuIpcMemHandle::Type::None;

  // POSIX FD handle
  if (cuMemExportToShareableHandle(&(handle->posixFd.fd), allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
      CUDA_SUCCESS) {
    handle->posixFd.pid = getpid();
    handle->typeFlags |= GpuIpcMemHandle::Type::PosixFd;
  }

  // FABRIC handle
  if (isFabricAvailable && (cuMemExportToShareableHandle(&(handle->fabric.handle), allocHandle,
                                                         CU_MEM_HANDLE_TYPE_FABRIC, 0) == CUDA_SUCCESS)) {
    handle->typeFlags |= GpuIpcMemHandle::Type::Fabric;
  }

  if (handle->typeFlags == GpuIpcMemHandle::Type::None) {
    throw Error("createMulticast failed: neither POSIX FD nor FABRIC handle was created", ErrorCode::SystemError);
  }
  return handle;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  throw Error("NVLS is not supported on this device (requires CUDA version >= 12.3 and Linux kernel version >= 5.6.0)",
              ErrorCode::InvalidUsage);
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

GpuIpcMem::GpuIpcMem(const GpuIpcMemHandle& handle)
    : handle_(handle),
      isMulticast_(false),
      multicastBindedAddr_(0),
      type_(GpuIpcMemHandle::Type::None),
      basePtr_(nullptr),
      baseSize_(0),
      dataPtr_(nullptr),
      dataSize_(0) {
  if (handle_.typeFlags == GpuIpcMemHandle::Type::None) {
    throw Error("GpuIpcMemHandle type is None, cannot create GpuIpcMem", ErrorCode::InvalidUsage);
  }
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
    cudaError_t err = cudaIpcOpenMemHandle(&basePtr_, handle_.runtimeIpc.handle, cudaIpcMemLazyEnablePeerAccess);
    if (err == cudaSuccess) {
      baseSize_ = handle_.baseSize;
      dataPtr_ = static_cast<void*>(static_cast<char*>(basePtr_) + handle_.offsetFromBase);
      dataSize_ = handle_.baseSize - handle_.offsetFromBase;
      type_ = GpuIpcMemHandle::Type::RuntimeIpc;
      return;
    } else {
      (void)cudaGetLastError();
    }
  }
  if (type_ == GpuIpcMemHandle::Type::None) {
    std::stringstream ss;
    ss << "Failed to open GpuIpcMemHandle (type: " << std::to_string(handle_.typeFlags) << ")";
    throw Error(ss.str(), ErrorCode::Aborted);
  }
}

GpuIpcMem::~GpuIpcMem() {
  if (type_ == GpuIpcMemHandle::Type::RuntimeIpc) {
    cudaError_t err = cudaIpcCloseMemHandle(basePtr_);
    if (err != cudaSuccess) {
      WARN("Failed to close CUDA IPC handle at pointer %p: %s", basePtr_, cudaGetErrorString(err));
      (void)cudaGetLastError();
    }
  } else if (type_ == GpuIpcMemHandle::Type::PosixFd || type_ == GpuIpcMemHandle::Type::Fabric) {
    CUresult res;
    const char* errStr;
    if (basePtr_) {
      res = cuMemUnmap((CUdeviceptr)basePtr_, baseSize_);
      if (res != CUDA_SUCCESS) {
        (void)cuGetErrorString(res, &errStr);
        WARN("Failed to unmap CUDA memory at pointer %p: %s", basePtr_, errStr);
      }
      res = cuMemAddressFree((CUdeviceptr)basePtr_, baseSize_);
      if (res != CUDA_SUCCESS) {
        (void)cuGetErrorString(res, &errStr);
        WARN("Failed to free CUDA memory at pointer %p: %s", basePtr_, errStr);
      }
    }
#if (CUDA_NVLS_API_AVAILABLE)
    if (isMulticast_ && multicastBindedAddr_) {
      int deviceId;
      res = cuPointerGetAttribute(&deviceId, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, multicastBindedAddr_);
      if (res != CUDA_SUCCESS) {
        (void)cuGetErrorString(res, &errStr);
        WARN("Failed to get device ordinal for pointer %p: %s", (void*)multicastBindedAddr_, errStr);
        deviceId = -1;
      } else if (deviceId < 0) {
        WARN("Invalid device ordinal %d for pointer %p", deviceId, (void*)multicastBindedAddr_);
      }
      CUdevice device;
      if (cuDeviceGet(&device, deviceId) == CUDA_SUCCESS) {
        (void)cuMulticastUnbind(allocHandle_, device, 0, baseSize_);
      }
    }
#endif  // (CUDA_NVLS_API_AVAILABLE)
    res = cuMemRelease(allocHandle_);
    if (res != CUDA_SUCCESS) {
      (void)cuGetErrorString(res, &errStr);
      WARN("Failed to release CUDA memory allocation handle: %s", errStr);
    }
  }
}

void* GpuIpcMem::map() {
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

void* GpuIpcMem::mapMulticast([[maybe_unused]] int numDevices, [[maybe_unused]] const CUdeviceptr bufferAddr,
                              [[maybe_unused]] size_t bufferSize) {
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
  if (bufferAddr != 0) {
    if (!isCuMemMapAllocated((void*)bufferAddr)) {
      throw Error("This NVLS connection tried to bind a buffer that was not allocated with cuMemMap",
                  ErrorCode::InvalidUsage);
    }
    if ((uintptr_t)bufferAddr % minMcGran != 0) {
      throw Error("This NVLS connection tried to bind a buffer that is not aligned to the minimum granularity",
                  ErrorCode::InvalidUsage);
    }
    if (bufferSize == 0) {
      throw Error("NVLS buffer size should be larger than zero.", ErrorCode::InvalidUsage);
    }
    if (bufferSize % minMcGran != 0) {
      std::stringstream ss;
      ss << "Tried to bind a multicast buffer that is not aligned to the minimum granularity " << minMcGran
         << ", buffer size: " << bufferSize;
      throw Error(ss.str(), ErrorCode::InvalidUsage);
    }
    bufferPtr = bufferAddr;
  } else {
    multicastBuffer_ = GpuBuffer<uint8_t>(handle_.baseSize).memory();
    bufferPtr = (CUdeviceptr)(multicastBuffer_.get());
    bufferSize = handle_.baseSize;
  }

  // will block until all devices call cuMulticastAddDevice()
  MSCCLPP_CUTHROW(cuMulticastBindAddr(allocHandle_, 0, bufferPtr, bufferSize, 0));
  multicastBindedAddr_ = bufferPtr;

  CUdeviceptr mcPtr;
  MSCCLPP_CUTHROW(cuMemAddressReserve(&mcPtr, bufferSize, minMcGran, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap(mcPtr, bufferSize, 0, allocHandle_, 0));

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(mcPtr, bufferSize, &accessDesc, 1));

  basePtr_ = (void*)mcPtr;
  baseSize_ = handle_.baseSize;
  dataPtr_ = basePtr_;
  dataSize_ = bufferSize;
  isMulticast_ = true;
  return dataPtr_;
#else   // !(CUDA_NVLS_API_AVAILABLE)
  throw Error("NVLS is not supported on this device (requires CUDA version >= 12.3 and Linux kernel version >= 5.6.0)",
              ErrorCode::InvalidUsage);
#endif  // !(CUDA_NVLS_API_AVAILABLE)
}

void* GpuIpcMem::data() const {
  if (!dataPtr_) {
    throw Error("GpuIpcMem data pointer is null. Call map() first.", ErrorCode::InvalidUsage);
  }
  return dataPtr_;
}

}  // namespace mscclpp
