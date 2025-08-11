// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_IPC_MEM_HPP_
#define MSCCLPP_GPU_IPC_MEM_HPP_

#include <bitset>
#include <memory>
#include <mscclpp/gpu.hpp>

namespace mscclpp {

struct GpuIpcMemHandle {
  struct Type {
    static constexpr uint8_t None = 0;
    static constexpr uint8_t RuntimeIpc = 1;
    static constexpr uint8_t PosixFd = 2;
    static constexpr uint8_t Fabric = 4;
  };

  uint8_t typeFlags;
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
  static std::unique_ptr<GpuIpcMemHandle, decltype(&GpuIpcMemHandle::deleter)> create(const CUdeviceptr ptr);
};

using UniqueGpuIpcMemHandle = std::unique_ptr<GpuIpcMemHandle, decltype(&GpuIpcMemHandle::deleter)>;

static_assert(std::is_trivially_copyable_v<GpuIpcMemHandle>);

class GpuIpcMem {
 public:
  GpuIpcMem(const GpuIpcMemHandle &handle);
  ~GpuIpcMem();

  void *data() const { return dataPtr_; }

 private:
  GpuIpcMemHandle handle_;
  uint8_t type_;
  void *basePtr_;
  size_t baseSize_;
  void *dataPtr_;
  size_t dataSize_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_IPC_MEM_HPP_
