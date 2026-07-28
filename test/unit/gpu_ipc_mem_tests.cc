// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <mscclpp/gpu_utils.hpp>

#include "../framework.hpp"
#include "gpu_ipc_mem.hpp"

#if CUDA_NVLS_API_AVAILABLE

namespace {

std::shared_ptr<uint64_t> allocatePhysicalMemory(size_t nElements) {
  size_t granularity = mscclpp::detail::getCuAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  return mscclpp::detail::gpuCallocPhysicalShared<uint64_t>(nElements, granularity, granularity);
}

void requirePosixFd(const mscclpp::GpuIpcMemHandle& handle) {
  if ((handle.typeFlags & mscclpp::GpuIpcMemHandle::Type::PosixFd) == 0) {
    SKIP_TEST() << "CUDA POSIX FD memory handles are unavailable";
  }
}

}  // namespace

TEST(GpuIpcMemTest, ReusesPosixFdForSameAllocation) {
  auto memory = allocatePhysicalMemory(2);
  auto baseHandle = mscclpp::GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(memory.get()));
  auto offsetHandle = mscclpp::GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(memory.get() + 1));
  requirePosixFd(*baseHandle);
  requirePosixFd(*offsetHandle);

  ASSERT_EQ(baseHandle->posixFd.fd, offsetHandle->posixFd.fd);
  EXPECT_EQ(offsetHandle->offsetFromBase - baseHandle->offsetFromBase, sizeof(uint64_t));

  int sharedFd = baseHandle->posixFd.fd;
  baseHandle.reset();
  EXPECT_NE(fcntl(sharedFd, F_GETFD), -1);

  offsetHandle.reset();
  errno = 0;
  EXPECT_EQ(fcntl(sharedFd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST(GpuIpcMemTest, UsesDifferentPosixFdsForDifferentAllocations) {
  auto firstMemory = allocatePhysicalMemory(1);
  auto secondMemory = allocatePhysicalMemory(1);
  auto firstHandle = mscclpp::GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(firstMemory.get()));
  auto secondHandle = mscclpp::GpuIpcMemHandle::create(reinterpret_cast<CUdeviceptr>(secondMemory.get()));
  requirePosixFd(*firstHandle);
  requirePosixFd(*secondHandle);

  EXPECT_NE(firstHandle->posixFd.fd, secondHandle->posixFd.fd);
}

#endif  // CUDA_NVLS_API_AVAILABLE
