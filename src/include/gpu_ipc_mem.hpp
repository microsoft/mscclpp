// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPU_IPC_MEM_HPP_
#define MSCCLPP_GPU_IPC_MEM_HPP_

#include <bitset>
#include <memory>
#include <ostream>

#include <mscclpp/gpu.hpp>

namespace mscclpp {

struct GpuIpcMemHandle {
  struct Type {
    static constexpr uint8_t None = 0;
    static constexpr uint8_t RuntimeIpc = 1;
    static constexpr uint8_t PosixFd = 2;
    static constexpr uint8_t Fabric = 4;
  };

  using TypeFlags = uint8_t;

  TypeFlags typeFlags;
  size_t baseSize;
  size_t offsetFromBase;

  struct {
    cudaIpcMemHandle_t handle;
  } runtimeIpc;

  struct {
    int pid;
    int fd;
  } posixFd;

  struct {
    char handle[64];
  } fabric;

  static void deleter(GpuIpcMemHandle *handle);

  // We make GpuIpcMemHandle trivially copyable for easy serialization,
  // and thus it cannot have explicit destructors.
  // We use a custom deleter for unique_ptr to handle cleanup without a destructor.
  struct UniquePtr : public std::unique_ptr<GpuIpcMemHandle, decltype(&GpuIpcMemHandle::deleter)> {
    using Base = std::unique_ptr<GpuIpcMemHandle, decltype(&GpuIpcMemHandle::deleter)>;

    // Default constructor
    UniquePtr() : Base(nullptr, &GpuIpcMemHandle::deleter) {}

    // Inherit other constructors
    using Base::Base;

    // Allow implicit conversion from Base
    UniquePtr(Base &&other) : Base(std::move(other)) {}
  };

  static UniquePtr create(const CUdeviceptr ptr);
  static UniquePtr createMulticast(size_t bufferSize, int numDevices);
};

using UniqueGpuIpcMemHandle = GpuIpcMemHandle::UniquePtr;

std::ostream& operator<<(std::ostream& os, const GpuIpcMemHandle::TypeFlags& typeFlags);

static_assert(std::is_trivially_copyable_v<GpuIpcMemHandle>);

class GpuIpcMem {
 public:
  GpuIpcMem(const GpuIpcMemHandle &handle);

  ~GpuIpcMem();

  void *map();

  void *mapMulticast(int numDevices, const CUdeviceptr bufferAddr = 0, size_t bufferSize = 0);

  void *multicastBuffer() const { return isMulticast_ ? multicastBuffer_.get() : nullptr; }

  void *data() const;

  size_t size() const { return dataSize_; }

 private:
  GpuIpcMemHandle handle_;
  CUmemGenericAllocationHandle allocHandle_;
  std::shared_ptr<uint8_t> multicastBuffer_;
  bool isMulticast_;
  CUdeviceptr multicastBindedAddr_;
  uint8_t type_;
  void *basePtr_;
  size_t baseSize_;
  void *dataPtr_;
  size_t dataSize_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_IPC_MEM_HPP_
