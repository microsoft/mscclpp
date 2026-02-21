// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GDR_HPP_
#define MSCCLPP_GDR_HPP_

namespace mscclpp {

/// Whether the global GDRCopy context is enabled.
bool gdrEnabled();

}  // namespace mscclpp

#ifdef MSCCLPP_USE_GDRCOPY

#include <gdrapi.h>

#include <cstddef>
#include <cstdint>
#include <memory>

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

  /// Copy data from host memory to the mapped GPU location.
  void copyTo(const void* src, size_t size);

 private:
  std::shared_ptr<GdrContext> ctx_;
  std::shared_ptr<void> gpuMem_;
  gdr_mh_t mh_{};
  void* barPtr_ = nullptr;
  volatile uint64_t* hostDstPtr_ = nullptr;
  size_t mappedSize_ = 0;
};

}  // namespace mscclpp

#endif  // MSCCLPP_USE_GDRCOPY
#endif  // MSCCLPP_GDR_HPP_
