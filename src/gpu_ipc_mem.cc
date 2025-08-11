// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu_ipc_mem.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <mscclpp/gpu_utils.hpp>
#include <sstream>

#include "debug.h"

namespace mscclpp {

static int dupFdFromPid(pid_t pid, int targetFd) {
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
  handle->offsetFromBase = (size_t)(ptr - basePtr);

  // Runtime IPC handle
  if (cudaIpcGetMemHandle(&handle->runtimeIpc.handle, (void*)basePtr) == cudaSuccess) {
    handle->typeFlags |= GpuIpcMemHandle::Type::RuntimeIpc;
  }

  CUmemGenericAllocationHandle allocHandle;
  CUresult res = cuMemRetainAllocationHandle(&allocHandle, (void*)basePtr);
  if (res == CUDA_ERROR_NOT_SUPPORTED) return handle;  // Not supported on this platform
  if (res == CUDA_ERROR_INVALID_VALUE) return handle;  // Not mapped by cuMem API
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

GpuIpcMem::GpuIpcMem(const GpuIpcMemHandle& handle) : handle_(handle) {
  CUmemGenericAllocationHandle allocHandle;
  type_ = GpuIpcMemHandle::Type::None;
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::Fabric)) {
    if (cuMemImportFromShareableHandle(&allocHandle, (void*)handle_.fabric.handle, CU_MEM_HANDLE_TYPE_FABRIC) ==
        CUDA_SUCCESS) {
      type_ = GpuIpcMemHandle::Type::Fabric;
    }
  }
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::PosixFd)) {
    int dupfd = dupFdFromPid(handle_.posixFd.pid, handle_.posixFd.fd);
    if (dupfd != -1) {
      if (cuMemImportFromShareableHandle(&allocHandle, (void*)(uintptr_t)dupfd,
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS) {
        type_ = GpuIpcMemHandle::Type::PosixFd;
      }
      close(dupfd);
    }
  }
  if ((type_ == GpuIpcMemHandle::Type::None) && (handle_.typeFlags & GpuIpcMemHandle::Type::RuntimeIpc)) {
    if (cudaIpcOpenMemHandle(&basePtr_, handle_.runtimeIpc.handle, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess) {
      baseSize_ = handle_.baseSize;
      dataPtr_ = static_cast<void*>(static_cast<char*>(basePtr_) + handle_.offsetFromBase);
      dataSize_ = handle_.baseSize - handle_.offsetFromBase;
      type_ = GpuIpcMemHandle::Type::RuntimeIpc;
      return;
    }
  }
  if (type_ == GpuIpcMemHandle::Type::None) {
    return;
  }

  // Continued from Fabric or PosixFd
  size_t pageSize = getpagesize();
  if (handle_.baseSize % pageSize) {
    std::stringstream ss;
    ss << "Tried to map remote GPU memory with size " << handle_.baseSize
       << " that is not a multiple of the local host page size " << pageSize;
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }
  CUdeviceptr base;
  size_t minGran = detail::getMulticastGranularity(handle_.baseSize, CU_MULTICAST_GRANULARITY_MINIMUM);
  size_t alignment = minGran;
  MSCCLPP_CUTHROW(cuMemAddressReserve(&base, handle_.baseSize, alignment, 0, 0));
  MSCCLPP_CUTHROW(cuMemMap(base, handle_.baseSize, 0, allocHandle, 0));

  CUmemAccessDesc accessDesc = {};
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess(base, handle_.baseSize, &accessDesc, 1));

  basePtr_ = (void*)base;
  baseSize_ = handle_.baseSize;
  dataPtr_ = static_cast<void*>(static_cast<char*>(basePtr_) + handle_.offsetFromBase);
  dataSize_ = handle_.baseSize - handle_.offsetFromBase;

  MSCCLPP_CUTHROW(cuMemRelease(allocHandle));
}

GpuIpcMem::~GpuIpcMem() {
  if (type_ == GpuIpcMemHandle::Type::RuntimeIpc) {
    cudaError_t err = cudaIpcCloseMemHandle(basePtr_);
    if (err != cudaSuccess) {
      WARN("Failed to close CUDA IPC handle at pointer %p: %s", basePtr_, cudaGetErrorString(err));
    }
  } else if (type_ == GpuIpcMemHandle::Type::PosixFd || type_ == GpuIpcMemHandle::Type::Fabric) {
    CUresult res;
    const char* errStr;
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
}

}  // namespace mscclpp
