// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gdr.hpp"

#if defined(MSCCLPP_USE_GDRCOPY)

#include <gdrapi.h>
#include <unistd.h>

#include <mscclpp/env.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "logger.hpp"

#ifndef GPU_PAGE_SHIFT
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK (~(GPU_PAGE_SIZE - 1))
#endif

namespace mscclpp {

// GdrContext

class GdrContext {
 public:
  GdrContext();
  ~GdrContext();

  GdrContext(const GdrContext&) = delete;
  GdrContext& operator=(const GdrContext&) = delete;

  GdrStatus status() const { return status_; }
  gdr_t handle() const { return handle_; }

 private:
  GdrStatus status_;
  gdr_t handle_;
};

static std::shared_ptr<GdrContext> gdrContext() {
  static auto instance = std::make_shared<GdrContext>();
  return instance;
}

GdrStatus gdrStatus() { return gdrContext()->status(); }

bool gdrEnabled() { return gdrStatus() == GdrStatus::Ok; }

const char* gdrStatusMessage() {
  switch (gdrStatus()) {
    case GdrStatus::Ok:
      return "GDRCopy initialized successfully";
    case GdrStatus::NotBuilt:
      return "mscclpp was not built with GDRCopy support (MSCCLPP_USE_GDRCOPY not set)";
    case GdrStatus::Disabled:
      return "GDRCopy is disabled via MSCCLPP_FORCE_DISABLE_GDR environment variable";
    case GdrStatus::DriverMissing:
      return "GDRCopy kernel driver is not loaded (/dev/gdrdrv not found)";
    case GdrStatus::OpenFailed:
      return "gdr_open() failed; GDRCopy driver may be misconfigured";
    default:
      return "unknown GDRCopy status";
  }
}

GdrContext::GdrContext() : status_(GdrStatus::Disabled), handle_(nullptr) {
  if (env()->forceDisableGdr) {
    INFO(GPU, "GDRCopy disabled via MSCCLPP_FORCE_DISABLE_GDR");
    status_ = GdrStatus::Disabled;
    return;
  }

  // Auto-detect: check if driver is available
  if (access("/dev/gdrdrv", F_OK) != 0) {
    INFO(GPU, "GDRCopy driver not detected, disabling GDRCopy");
    status_ = GdrStatus::DriverMissing;
    return;
  }

  handle_ = gdr_open();
  if (handle_ == nullptr) {
    INFO(GPU, "gdr_open() failed, disabling GDRCopy");
    status_ = GdrStatus::OpenFailed;
    return;
  }

  status_ = GdrStatus::Ok;
  INFO(GPU, "GDRCopy initialized successfully");
}

GdrContext::~GdrContext() {
  if (handle_ != nullptr) {
    gdr_close(handle_);
    handle_ = nullptr;
  }
}

// GdrMap::Impl — real implementation with GDRCopy

struct GdrMap::Impl {
  std::shared_ptr<GdrContext> ctx;
  std::shared_ptr<void> gpuMem;
  gdr_mh_t mh;
  void* barPtr;
  uint64_t* hostDstPtr;
  size_t mappedSize;
};

GdrMap::GdrMap(std::shared_ptr<void> gpuMem, int deviceId) : pimpl_(std::make_unique<Impl>()) {
  pimpl_->ctx = gdrContext();
  pimpl_->gpuMem = std::move(gpuMem);
  pimpl_->mh = {};
  pimpl_->barPtr = nullptr;
  pimpl_->hostDstPtr = nullptr;
  pimpl_->mappedSize = 0;

  // Ensure CUDA device context is active for gdr_pin_buffer
  CudaDeviceGuard deviceGuard(deviceId);

  uint64_t gpuAddr = reinterpret_cast<uint64_t>(pimpl_->gpuMem.get());
  // Align to GPU page boundary and pin one page around the target address
  unsigned long alignedAddr = gpuAddr & GPU_PAGE_MASK;
  unsigned long pageOffset = gpuAddr - alignedAddr;
  pimpl_->mappedSize = GPU_PAGE_SIZE;

  // Pin the GPU memory for GDRCopy BAR1 mapping. Try GDR_PIN_FLAG_FORCE_PCIE first for optimal
  // ordering on platforms that support it (e.g., GB200). Fall back to flags=0 if FORCE_PCIE is
  // not supported. Both paths work correctly: CPU writes via atomicStore, GPU reads via
  // system-scope acquire.
  int ret =
      gdr_pin_buffer_v2(pimpl_->ctx->handle(), alignedAddr, pimpl_->mappedSize, GDR_PIN_FLAG_FORCE_PCIE, &pimpl_->mh);
  if (ret != 0) {
    ret = gdr_pin_buffer_v2(pimpl_->ctx->handle(), alignedAddr, pimpl_->mappedSize, 0, &pimpl_->mh);
    if (ret != 0) {
      THROW(GPU, Error, ErrorCode::InternalError, "gdr_pin_buffer_v2 failed (ret=", ret, ") for addr ", (void*)gpuAddr,
            ". Ensure the GPU memory is allocated with cudaMalloc (not cuMemCreate/cuMemMap).");
    }
  }

  ret = gdr_map(pimpl_->ctx->handle(), pimpl_->mh, &pimpl_->barPtr, pimpl_->mappedSize);
  if (ret != 0) {
    (void)gdr_unpin_buffer(pimpl_->ctx->handle(), pimpl_->mh);
    THROW(GPU, Error, ErrorCode::InternalError, "gdr_map failed (ret=", ret, ") for addr ", (void*)gpuAddr);
  }

  pimpl_->hostDstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(pimpl_->barPtr) + pageOffset);

  INFO(GPU, "GDRCopy mapping established: GPU addr ", (void*)gpuAddr, " -> host ptr ", (const void*)pimpl_->hostDstPtr);
}

GdrMap::~GdrMap() {
  if (pimpl_) {
    if (pimpl_->barPtr != nullptr) {
      (void)gdr_unmap(pimpl_->ctx->handle(), pimpl_->mh, pimpl_->barPtr, pimpl_->mappedSize);
    }
    if (pimpl_->hostDstPtr != nullptr) {
      (void)gdr_unpin_buffer(pimpl_->ctx->handle(), pimpl_->mh);
    }
  }
}

bool GdrMap::valid() const { return pimpl_ && pimpl_->hostDstPtr != nullptr; }

uint64_t* GdrMap::hostPtr() const { return pimpl_ ? pimpl_->hostDstPtr : nullptr; }

void GdrMap::copyTo(const void* src, size_t size) { gdr_copy_to_mapping(pimpl_->mh, pimpl_->hostDstPtr, src, size); }

void GdrMap::copyFrom(void* dst, size_t size) const {
  gdr_copy_from_mapping(pimpl_->mh, dst, pimpl_->hostDstPtr, size);
}

}  // namespace mscclpp

#else  // !defined(MSCCLPP_USE_GDRCOPY)

namespace mscclpp {

GdrStatus gdrStatus() { return GdrStatus::NotBuilt; }

bool gdrEnabled() { return false; }

const char* gdrStatusMessage() { return "mscclpp was not built with GDRCopy support (MSCCLPP_USE_GDRCOPY not set)"; }

// GdrMap::Impl — stub (no GDRCopy)

struct GdrMap::Impl {};

GdrMap::GdrMap(std::shared_ptr<void> /*gpuMem*/, int /*deviceId*/) {}

GdrMap::~GdrMap() = default;

bool GdrMap::valid() const { return false; }

uint64_t* GdrMap::hostPtr() const { return nullptr; }

void GdrMap::copyTo(const void* /*src*/, size_t /*size*/) {}

void GdrMap::copyFrom(void* /*dst*/, size_t /*size*/) const {}

}  // namespace mscclpp

#endif  // !defined(MSCCLPP_USE_GDRCOPY)
