// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>

#include "api.cuh"
#include "ht_runtime.hpp"
#include "ll_runtime.hpp"

namespace mscclpp {
namespace ep {

/// Single entry point for both MoE backends.
///
/// `MoEMode` selects which implementation is constructed; only that backend's
/// methods are usable afterwards. The two backends expose genuinely different
/// call protocols -- low latency is dispatch/combine, high throughput is
/// layout -> notifyDispatch -> resolveRecvXBuffer -> dispatch -- so the
/// forwarding methods stay prefixed by mode rather than pretending to share one
/// signature. Calling a method belonging to the other mode throws.
class MoERuntime {
 public:
  /// Construct the backend selected by @p mode.
  /// Low-latency uses @p maxTokensPerRank, @p hidden, @p numExperts and @p numTopk;
  /// high-throughput uses @p maxHiddenBytes and @p numSms.
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, int64_t maxHiddenBytes, int numSms);
  ~MoERuntime();

  MoEMode mode() const { return mode_; }
  bool isAvailable() const;
  bool isInternodeAvailable() const;

  /// @name Low-latency path
  /// @{
  MoELowLatencyRuntime& lowLatency();
  const MoELowLatencyRuntime& lowLatency() const;
  /// @}

  /// @name High-throughput path
  /// @{
  MoEHighThroughputRuntime& highThroughput();
  const MoEHighThroughputRuntime& highThroughput() const;
  /// @}

 private:
  MoEMode mode_;
  std::unique_ptr<MoELowLatencyRuntime> lowLatency_;
  std::unique_ptr<MoEHighThroughputRuntime> highThroughput_;
};

}  // namespace ep
}  // namespace mscclpp
