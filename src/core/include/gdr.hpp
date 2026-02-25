// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GDR_HPP_
#define MSCCLPP_GDR_HPP_

namespace mscclpp {

enum class GdrStatus {
  Ok,             // GDRCopy initialized successfully
  NotBuilt,       // Built without MSCCLPP_USE_GDRCOPY
  Disabled,       // Disabled via MSCCLPP_FORCE_DISABLE_GDR
  DriverMissing,  // /dev/gdrdrv not found
  OpenFailed,     // gdr_open() failed
};

/// Return the detailed status of the global GDRCopy context.
GdrStatus gdrStatus();

/// Whether the global GDRCopy context is enabled (shorthand for gdrStatus() == GdrStatus::Ok).
bool gdrEnabled();

}  // namespace mscclpp

#include <cstddef>
#include <cstdint>
#include <memory>

#if defined(MSCCLPP_USE_GDRCOPY)

#include <gdrapi.h>

namespace mscclpp {

class GdrContext;

/// RAII wrapper for a per-connection GDRCopy BAR1 mapping of a GPU address.
class GdrMap {
 public:
  /// Pin and map a GPU address for direct host-side access.
  /// Holds a shared reference to the GPU memory to keep it alive.
  /// @param gpuMem   Shared pointer to the GPU memory (e.g. from gpuCallocShared).
  /// @param deviceId The CUDA device ID for setting context.
  GdrMap(std::shared_ptr<void> gpuMem, int deviceId);
  ~GdrMap();

  GdrMap(const GdrMap&) = delete;
  GdrMap& operator=(const GdrMap&) = delete;

  /// Whether the mapping was established successfully.
  bool valid() const { return hostDstPtr_ != nullptr; }

  /// Return the BAR1-mapped host pointer to the GPU location.
  uint64_t* hostPtr() const { return hostDstPtr_; }

  /// Copy data from host memory to the mapped GPU location.
  void copyTo(const void* src, size_t size);

  /// Copy data from the mapped GPU location to host memory.
  void copyFrom(void* dst, size_t size) const;

 private:
  std::shared_ptr<GdrContext> ctx_;
  std::shared_ptr<void> gpuMem_;
  gdr_mh_t mh_;
  void* barPtr_;
  uint64_t* hostDstPtr_;
  size_t mappedSize_;
};

}  // namespace mscclpp

#else  // !defined(MSCCLPP_USE_GDRCOPY)

namespace mscclpp {

/// Stub GdrMap when GDRCopy is not available.
class GdrMap {
 public:
  GdrMap(std::shared_ptr<void> /*gpuMem*/, int /*deviceId*/) {}
  ~GdrMap() = default;

  GdrMap(const GdrMap&) = delete;
  GdrMap& operator=(const GdrMap&) = delete;

  bool valid() const { return false; }
  void copyTo(const void* /*src*/, size_t /*size*/) {}
  void copyFrom(void* /*dst*/, size_t /*size*/) const {}
  uint64_t* hostPtr() const { return nullptr; }
};

}  // namespace mscclpp

#endif  // !defined(MSCCLPP_USE_GDRCOPY)
#endif  // MSCCLPP_GDR_HPP_
