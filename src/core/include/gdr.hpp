// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GDR_HPP_
#define MSCCLPP_GDR_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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

/// Return a human-readable error message for the current GDRCopy status.
std::string gdrStatusMessage();

/// RAII wrapper for a GDRCopy BAR1 mapping of a GPU address.
/// When GDRCopy is not available, all operations are no-ops and valid() returns false.
class GdrMap {
 public:
  /// Pin and map a GPU address for direct host-side access.
  /// @param gpuMem   Shared pointer to the GPU memory (e.g. from gpuCallocShared).
  /// @param deviceId The CUDA device ID for setting context.
  GdrMap(std::shared_ptr<void> gpuMem, int deviceId);
  ~GdrMap();

  GdrMap(const GdrMap&) = delete;
  GdrMap& operator=(const GdrMap&) = delete;

  /// Whether the mapping was established successfully.
  bool valid() const;

  /// Return the BAR1-mapped host pointer to the GPU location.
  uint64_t* hostPtr() const;

  /// Copy data from host memory to the mapped GPU location.
  void copyTo(const void* src, size_t size);

  /// Copy data from the mapped GPU location to host memory.
  void copyFrom(void* dst, size_t size) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GDR_HPP_
