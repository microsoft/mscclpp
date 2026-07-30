// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <memory>
#include <mscclpp/core.hpp>

#include "api.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

/// Common interface and shared resources for every MoE backend.
///
/// Both the low-latency and high-throughput runtimes derive the same rank
/// topology from the bootstrap and report availability the same way; this base
/// owns that logic so the backends only implement their own algorithm. It is
/// also the polymorphic handle callers hold: `createMoERuntime` returns one of
/// these, and `mode()` identifies which backend is behind it.
class MoERuntime {
 public:
  // The backends release CUDA resources in their destructors and surface
  // failures, so the base destructor must permit exceptions too.
  virtual ~MoERuntime() noexcept(false) {}

  MoERuntime(const MoERuntime&) = delete;
  MoERuntime& operator=(const MoERuntime&) = delete;

  /// Backend this runtime implements.
  virtual MoEMode mode() const = 0;

  bool isAvailable() const { return available_; }
  bool isInternodeAvailable() const { return isAvailable() && numRanks_ > numNvlRanks_; }

  int rank() const { return rank_; }
  int numRanks() const { return numRanks_; }
  int numNvlRanks() const { return numNvlRanks_; }
  int numRanksPerIpcDomain() const { return numRanksPerIpcDomain_; }

 protected:
  explicit MoERuntime(mscclpp::Communicator& communicator)
      : bootstrap_(communicator.bootstrap()),
        rank_(bootstrap_->getRank()),
        numRanks_(bootstrap_->getNranks()),
        numNvlRanks_(std::min(numRanks_, bootstrap_->getNranksPerNode())),
        numRanksPerIpcDomain_(std::max(numNvlRanks_, std::min(numRanks_, bootstrap_->getNranksPerIpcDomain()))) {
    EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
    EP_HOST_ASSERT(numNvlRanks_ > 0);
  }

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
};

}  // namespace ep
}  // namespace mscclpp
