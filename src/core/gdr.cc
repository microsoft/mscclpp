// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gdr.hpp"

#ifdef MSCCLPP_USE_GDRCOPY

#include <unistd.h>

#include <mscclpp/env.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "logger.hpp"

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK (~(GPU_PAGE_SIZE - 1))

namespace mscclpp {

// GdrContext

class GdrContext {
 public:
  GdrContext();
  ~GdrContext();

  GdrContext(const GdrContext&) = delete;
  GdrContext& operator=(const GdrContext&) = delete;

  bool enabled() const { return enabled_; }
  gdr_t handle() const { return handle_; }

 private:
  bool enabled_ = false;
  gdr_t handle_ = nullptr;
};

static std::shared_ptr<GdrContext> gdrContext() {
  static auto instance = std::make_shared<GdrContext>();
  return instance;
}

bool gdrEnabled() { return gdrContext()->enabled(); }

GdrContext::GdrContext() {
  if (env()->forceDisableGdr) {
    INFO(GPU, "GDRCopy disabled via MSCCLPP_FORCE_DISABLE_GDR");
    return;
  }

  // Auto-detect: check if driver is available
  if (access("/dev/gdrdrv", F_OK) != 0) {
    INFO(GPU, "GDRCopy driver not detected, disabling GDRCopy");
    return;
  }

  handle_ = gdr_open();
  if (handle_ == nullptr) {
    INFO(GPU, "gdr_open() failed, disabling GDRCopy");
    return;
  }

  enabled_ = true;
  INFO(GPU, "GDRCopy initialized successfully");
}

GdrContext::~GdrContext() {
  if (handle_ != nullptr) {
    gdr_close(handle_);
    handle_ = nullptr;
  }
}

// GdrMap

GdrMap::GdrMap(std::shared_ptr<void> gpuMem, int deviceId) : ctx_(gdrContext()), gpuMem_(std::move(gpuMem)) {
  // Ensure CUDA device context is active for gdr_pin_buffer
  CudaDeviceGuard deviceGuard(deviceId);

  uint64_t gpuAddr = reinterpret_cast<uint64_t>(gpuMem_.get());
  // Align to GPU page boundary and pin one page around the target address
  unsigned long alignedAddr = gpuAddr & GPU_PAGE_MASK;
  unsigned long pageOffset = gpuAddr - alignedAddr;
  mappedSize_ = GPU_PAGE_SIZE;

  int ret = gdr_pin_buffer(ctx_->handle(), alignedAddr, mappedSize_, 0, 0, &mh_);
  if (ret != 0) {
    THROW(GPU, Error, ErrorCode::InternalError, "gdr_pin_buffer failed (ret=", ret, ") for addr ", (void*)gpuAddr,
          ". Ensure the GPU memory is allocated with cudaMalloc (not cuMemCreate/cuMemMap).");
  }

  ret = gdr_map(ctx_->handle(), mh_, &barPtr_, mappedSize_);
  if (ret != 0) {
    (void)gdr_unpin_buffer(ctx_->handle(), mh_);
    THROW(GPU, Error, ErrorCode::InternalError, "gdr_map failed (ret=", ret, ") for addr ", (void*)gpuAddr);
  }

  hostDstPtr_ = reinterpret_cast<volatile uint64_t*>(reinterpret_cast<char*>(barPtr_) + pageOffset);

  INFO(GPU, "GDRCopy mapping established: GPU addr ", (void*)gpuAddr, " -> host ptr ", (const void*)hostDstPtr_);
}

GdrMap::~GdrMap() {
  if (barPtr_ != nullptr) {
    (void)gdr_unmap(ctx_->handle(), mh_, barPtr_, mappedSize_);
  }
  if (hostDstPtr_ != nullptr) {
    (void)gdr_unpin_buffer(ctx_->handle(), mh_);
  }
}

void GdrMap::copyTo(const void* src, size_t size) { gdr_copy_to_mapping(mh_, (void*)hostDstPtr_, src, size); }

}  // namespace mscclpp

#else  // !MSCCLPP_USE_GDRCOPY

namespace mscclpp {

bool gdrEnabled() { return false; }

}  // namespace mscclpp

#endif  // MSCCLPP_USE_GDRCOPY
