// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "registered_memory.hpp"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <mscclpp/gpu_utils.hpp>
#include <unordered_map>

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
      WARN(mscclpp::P2P, "Call to " #cmd " failed, error is ", errStr); \
    }                                                                   \
  } while (false)

namespace {

// Custom hash and equality for cudaIpcMemHandle_t
struct CudaIpcMemHandleHash {
  size_t operator()(const cudaIpcMemHandle_t& handle) const {
    std::string_view view(handle.reserved, sizeof(handle.reserved));
    return std::hash<std::string_view>{}(view);
  }
};

struct CudaIpcMemHandleEqual {
  bool operator()(const cudaIpcMemHandle_t& lhs, const cudaIpcMemHandle_t& rhs) const noexcept {
    return std::memcmp(lhs.reserved, rhs.reserved, sizeof(lhs.reserved)) == 0;
  }
};

CUmemAllocationHandleType getNvlsMemHandleType() {
#if (CUDA_NVLS_API_AVAILABLE)
  if (mscclpp::detail::nvlsCompatibleMemHandleType & CU_MEM_HANDLE_TYPE_FABRIC) {
    return CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
#else
  throw mscclpp::Error("Only support GPU with NVLS support", mscclpp::ErrorCode::InvalidUsage);
#endif
}

std::shared_ptr<void> getPeerMemoryHandle(cudaIpcMemHandle_t ipcHandle) {
  void* addr;
  auto deleter = [](void* p) {
    cudaError_t err = cudaIpcCloseMemHandle(p);
    if (err != cudaSuccess) {
      WARN(mscclpp::P2P, "Failed to close CUDA IPC handle at pointer ", std::hex, p, ": ", cudaGetErrorString(err));
    } else {
      INFO(mscclpp::P2P, "Closed CUDA IPC handle at pointer ", std::hex, p);
    }
  };
#if defined(MSCCLPP_USE_ROCM)
  // Unlike Nvidia, ROCm will not reuse the same ipc handle for same memory region.
  // We cache the opened ipc handles to avoid opening multiple times. (May exceed system limit on vm.max_map_count)
  static auto peerMemoryHandleMap = std::make_shared<
      std::unordered_map<cudaIpcMemHandle_t, std::weak_ptr<void>, CudaIpcMemHandleHash, CudaIpcMemHandleEqual>>();
  static auto mutex = std::make_shared<std::mutex>();
  std::lock_guard<std::mutex> lock(*mutex);
  auto it = peerMemoryHandleMap->find(ipcHandle);
  if (it != peerMemoryHandleMap->end()) {
    if (auto ptr = it->second.lock()) {
      return ptr;
    } else {
      peerMemoryHandleMap->erase(it);
    }
  }
  MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(&addr, ipcHandle, cudaIpcMemLazyEnablePeerAccess));
  std::shared_ptr<void> ptr =
      std::shared_ptr<void>(addr, [ipcHandle, deleter, m = mutex, map = peerMemoryHandleMap](void* p) {
        deleter(p);
        std::lock_guard<std::mutex> lock(*m);
        map->erase(ipcHandle);
      });
  peerMemoryHandleMap->emplace(ipcHandle, ptr);
  return ptr;
#else
  MSCCLPP_CUDATHROW(cudaIpcOpenMemHandle(&addr, ipcHandle, cudaIpcMemLazyEnablePeerAccess));
  return std::shared_ptr<void>(addr, deleter);
#endif
}

}  // namespace

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

    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;

    void* baseDataPtr;
    size_t baseDataSize;
    MSCCLPP_CUTHROW(cuMemGetAddressRange((CUdeviceptr*)&baseDataPtr, &baseDataSize, (CUdeviceptr)data));
    this->baseDataSize = baseDataSize;
    this->isCuMemMapAlloc = isCuMemMapAllocated(baseDataPtr);
    if (this->isCuMemMapAlloc) {
      CUmemGenericAllocationHandle handle;
      MSCCLPP_CUTHROW(cuMemRetainAllocationHandle(&handle, baseDataPtr));
      if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
        MSCCLPP_CUTHROW(cuMemExportToShareableHandle(transportInfo.shareableHandle, handle, getNvlsMemHandleType(), 0));
      } else {
        MSCCLPP_CUTHROW(cuMemExportToShareableHandle(&this->fileDesc, handle, getNvlsMemHandleType(), 0));
        transportInfo.rootFd = UnixSocketServer::instance().registerFd(fileDesc);
        transportInfo.rootPid = getpid();
      }
      transportInfo.offsetFromBase = (char*)data - (char*)baseDataPtr;
      MSCCLPP_CUTHROW(cuMemRelease(handle));
    } else {
      cudaIpcMemHandle_t handle;
      MSCCLPP_CUDATHROW(cudaIpcGetMemHandle(&handle, baseDataPtr));
      // TODO: bug with offset of base?
      transportInfo.cudaIpcBaseHandle = handle;
      transportInfo.cudaIpcOffsetFromBase = (char*)data - (char*)baseDataPtr;
    }
    this->transportInfos.push_back(transportInfo);
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
      INFO(NET, "IB mr for address ", data, " with size ", size, " is registered");
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
  detail::serialize(result, pimpl_->baseDataSize);
  detail::serialize(result, pimpl_->hostHash);
  detail::serialize(result, pimpl_->pidHash);
  detail::serialize(result, pimpl_->isCuMemMapAlloc);
  detail::serialize(result, pimpl_->transports);
  if (pimpl_->transportInfos.size() > static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw Error("Too many transport info entries", ErrorCode::InternalError);
  }
  int8_t transportCount = pimpl_->transportInfos.size();
  detail::serialize(result, transportCount);
  for (auto& entry : pimpl_->transportInfos) {
    detail::serialize(result, entry.transport);
    if (entry.transport == Transport::CudaIpc) {
      if (pimpl_->isCuMemMapAlloc) {
        if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
          detail::serialize(result, entry.shareableHandle);
        } else {
          detail::serialize(result, entry.rootFd);
          detail::serialize(result, entry.rootPid);
        }
        detail::serialize(result, entry.offsetFromBase);
      } else {
        detail::serialize(result, entry.cudaIpcBaseHandle);
        detail::serialize(result, entry.cudaIpcOffsetFromBase);
      }
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
  it = detail::deserialize(it, this->baseDataSize);
  it = detail::deserialize(it, this->hostHash);
  it = detail::deserialize(it, this->pidHash);
  it = detail::deserialize(it, this->isCuMemMapAlloc);
  it = detail::deserialize(it, this->transports);
  int8_t transportCount;
  it = detail::deserialize(it, transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    it = detail::deserialize(it, transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      if (this->isCuMemMapAlloc) {
        if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
          it = detail::deserialize(it, transportInfo.shareableHandle);
        } else {
          it = detail::deserialize(it, transportInfo.rootFd);
          it = detail::deserialize(it, transportInfo.rootPid);
        }
        it = detail::deserialize(it, transportInfo.offsetFromBase);
      } else {
        it = detail::deserialize(it, transportInfo.cudaIpcBaseHandle);
        it = detail::deserialize(it, transportInfo.cudaIpcOffsetFromBase);
      }
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
    if (this->isCuMemMapAlloc) {
      // Query which device owns this memory
      int gpuId = detail::gpuIdFromAddress(this->data);
      int currentDevice = -1;
      MSCCLPP_CUDATHROW(cudaGetDevice(&currentDevice));

      // Only set access if we're on a different device than where memory was allocated
      if (gpuId != currentDevice) {
        detail::setReadWriteMemoryAccess(this->data, this->baseDataSize);
      }
    }
  } else if (transports.has(Transport::CudaIpc)) {
    // The memory is local to the machine but not to the process, so we need to open the CUDA IPC handle
    auto entry = getTransportInfo(Transport::CudaIpc);
    void* base;
    if (this->isCuMemMapAlloc) {
#if (CUDA_NVLS_API_AVAILABLE)
      CUmemGenericAllocationHandle handle;
      if (getHostHash() != this->hostHash) {
        // TODO: only open handle if in same MNNVL domain
        CUresult err = cuMemImportFromShareableHandle(&handle, entry.shareableHandle, getNvlsMemHandleType());
        if (err != CUDA_SUCCESS) {
          INFO(P2P, "Failed to import shareable handle from host: 0x", std::hex, hostHash,
               ", may not be in the same MNNVL domain");
          return;
        }
      } else {
        if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
          MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&handle, entry.shareableHandle, getNvlsMemHandleType()));
        } else {
          int fd =
              UnixSocketClient::instance().requestFd(UnixSocketServer::generateSocketPath(entry.rootPid), entry.rootFd);
          INFO(P2P, "Get file descriptor ", fd, " from peer 0x", std::hex, hostHash);
          MSCCLPP_CUTHROW(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(fd),
                                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
          close(fd);
        }
      }
      size_t minGran = detail::getMulticastGranularity(this->baseDataSize, CU_MULTICAST_GRANULARITY_MINIMUM);
      size_t recommendedGran =
          detail::getMulticastGranularity(this->baseDataSize, CU_MULTICAST_GRANULARITY_RECOMMENDED);
      size_t size = (this->baseDataSize + recommendedGran - 1) / recommendedGran * recommendedGran;
      MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)&base, size, minGran, 0, 0));
      MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)base, size, 0, handle, 0));
      detail::setReadWriteMemoryAccess(base, size);
      this->data = static_cast<char*>(base) + entry.offsetFromBase;
#else   // !(CUDA_NVLS_API_AVAILABLE)
      // never reach here.
      throw Error("Unexpected error", ErrorCode::InternalError);
#endif  // !(CUDA_NVLS_API_AVAILABLE)
    } else if (getHostHash() == this->hostHash) {
      this->peerMemHandle = getPeerMemoryHandle(entry.cudaIpcBaseHandle);
      this->data = static_cast<char*>(this->peerMemHandle.get()) + entry.cudaIpcOffsetFromBase;
    }
  }
  if (this->data != nullptr) {
    INFO(P2P, "Opened CUDA IPC handle at pointer ", this->data);
  }
}

RegisteredMemory::Impl::Impl(const std::vector<char>& serialization)
    : Impl(serialization.begin(), serialization.end()) {}

RegisteredMemory::Impl::~Impl() {
  // Close the CUDA IPC handle if it was opened during deserialization or initialization
  if (data && transports.has(Transport::CudaIpc) && getHostHash() == this->hostHash) {
    if (getPidHash() == this->pidHash) {
      // For local registered memory
      if (fileDesc >= 0) {
        close(fileDesc);
        UnixSocketServer::instance().unregisterFd(fileDesc);
        fileDesc = -1;
      }
      return;
    }
    // For remote registered memory
    void* base = static_cast<char*>(data) - getTransportInfo(Transport::CudaIpc).cudaIpcOffsetFromBase;
    if (this->isCuMemMapAlloc) {
      CUmemGenericAllocationHandle handle;
      size_t size = 0;
      MSCCLPP_CULOG_WARN(cuMemRetainAllocationHandle(&handle, base));
      MSCCLPP_CULOG_WARN(cuMemRelease(handle));
      MSCCLPP_CULOG_WARN(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)base));
      MSCCLPP_CULOG_WARN(cuMemUnmap((CUdeviceptr)base, size));
      MSCCLPP_CULOG_WARN(cuMemRelease(handle));
      MSCCLPP_CULOG_WARN(cuMemAddressFree((CUdeviceptr)base, size));
    }
    data = nullptr;
    fileDesc = -1;
  }
}

const TransportInfo& RegisteredMemory::Impl::getTransportInfo(Transport transport) const {
  for (auto& entry : transportInfos) {
    if (entry.transport == transport) {
      return entry;
    }
  }
  throw Error("Transport data not found", ErrorCode::InternalError);
}

}  // namespace mscclpp
