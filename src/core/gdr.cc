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

// mscclpp's GDRCopy path uses gdr_pin_buffer_v2, which was added to the gdrdrv kernel module
// in 2.5. Older modules return ENOTTY for the v2 ioctl, surfacing as a confusing "ret=25"
// failure deep inside GdrMap. Refuse early with a clear status when the loaded module is older.
#define MSCCLPP_GDRDRV_MIN_MAJOR 2
#define MSCCLPP_GDRDRV_MIN_MINOR 5

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
  int driverMajor() const { return driverMajor_; }
  int driverMinor() const { return driverMinor_; }

 private:
  GdrStatus status_;
  gdr_t handle_;
  int driverMajor_;
  int driverMinor_;
};

static std::shared_ptr<GdrContext> gdrContext() {
  static auto instance = std::make_shared<GdrContext>();
  return instance;
}

GdrStatus gdrStatus() { return gdrContext()->status(); }

bool gdrEnabled() { return gdrStatus() == GdrStatus::Ok; }

std::string gdrStatusMessage() {
  auto ctx = gdrContext();
  switch (ctx->status()) {
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
    case GdrStatus::KernelTooOld:
      return "gdrdrv kernel module " + std::to_string(ctx->driverMajor()) + "." + std::to_string(ctx->driverMinor()) +
             " is older than the required minimum (" + std::to_string(MSCCLPP_GDRDRV_MIN_MAJOR) + "." +
             std::to_string(MSCCLPP_GDRDRV_MIN_MINOR) + "); reinstall gdrcopy (e.g. v2.5.2) on the host";
    default:
      return "unknown GDRCopy status";
  }
}

GdrContext::GdrContext() : status_(GdrStatus::Disabled), handle_(nullptr), driverMajor_(0), driverMinor_(0) {
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

  // Reject kernel modules older than the minimum required for gdr_pin_buffer_v2.
  // Without this, GdrMap would later fail deep inside the v2 ioctl with ENOTTY (ret=25).
  if (gdr_driver_get_version(handle_, &driverMajor_, &driverMinor_) != 0) {
    INFO(GPU, "gdr_driver_get_version() failed; cannot verify kernel module version, disabling GDRCopy");
    gdr_close(handle_);
    handle_ = nullptr;
    status_ = GdrStatus::KernelTooOld;
    return;
  }
  if (driverMajor_ < MSCCLPP_GDRDRV_MIN_MAJOR ||
      (driverMajor_ == MSCCLPP_GDRDRV_MIN_MAJOR && driverMinor_ < MSCCLPP_GDRDRV_MIN_MINOR)) {
    WARN(GPU, "gdrdrv kernel module ", driverMajor_, ".", driverMinor_, " predates the v2 pin-buffer ioctl (need ",
         MSCCLPP_GDRDRV_MIN_MAJOR, ".", MSCCLPP_GDRDRV_MIN_MINOR, "+); disabling GDRCopy");
    gdr_close(handle_);
    handle_ = nullptr;
    status_ = GdrStatus::KernelTooOld;
    return;
  }

  int libMajor = 0, libMinor = 0;
  gdr_runtime_get_version(&libMajor, &libMinor);
  status_ = GdrStatus::Ok;
  INFO(GPU, "GDRCopy initialized: libgdrapi ", libMajor, ".", libMinor, ", gdrdrv ", driverMajor_, ".", driverMinor_);
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
      // ENOTTY (25) here means the loaded gdrdrv kernel module doesn't recognise the v2 ioctl
      // — GdrContext's version gate normally catches that earlier, so reaching here implies
      // a real allocator or driver problem.
      THROW(GPU, Error, ErrorCode::InternalError, "gdr_pin_buffer_v2 failed (ret=", ret, ") for addr ", (void*)gpuAddr,
            "; gdrdrv ", pimpl_->ctx->driverMajor(), ".", pimpl_->ctx->driverMinor(),
            ". If ret==25 (ENOTTY), the kernel module is too old; otherwise ensure the GPU memory is "
            "allocated with cudaMalloc (not cuMemCreate/cuMemMap).");
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

std::string gdrStatusMessage() { return "mscclpp was not built with GDRCopy support (MSCCLPP_USE_GDRCOPY not set)"; }

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
