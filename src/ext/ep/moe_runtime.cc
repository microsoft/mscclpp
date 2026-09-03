// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <stdexcept>

#include "exception.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden,
                       int numExperts, int numTopk, DispatchLayout outputLayout, CombineMode combineMode)
    : bootstrap_(communicator.bootstrap()),
      mode_(mode),
      rank_(bootstrap_->getRank()),
      numRanks_(bootstrap_->getNranks()),
      numNvlRanks_(std::min(numRanks_, bootstrap_->getNranksPerNode())),
      numRanksPerIpcDomain_(std::max(numNvlRanks_, std::min(numRanks_, bootstrap_->getNranksPerIpcDomain()))) {
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
  EP_HOST_ASSERT(numNvlRanks_ > 0);

  if (mode_ != MoEMode::LATENCY) {
    throw std::invalid_argument("This build only supports MoEMode::LATENCY");
  }
  latencyContext_ =
      std::make_unique<LatencyRuntimeContext>(communicator, rank_, numRanks_, numNvlRanks_, numRanksPerIpcDomain_,
                                              maxTokensPerRank, hidden, numExperts, numTopk, outputLayout, combineMode);
  available_ = latencyContext_->available_;
}

MoERuntime::~MoERuntime() noexcept(false) = default;

void MoERuntime::requireMode(MoEMode expected) const {
  if (mode_ != expected) {
    throw std::runtime_error(expected == MoEMode::LATENCY ? "MoE runtime was not created with MoEMode::LATENCY"
                                                          : "MoE runtime was not created with MoEMode::THROUGHPUT");
  }
}

void MoERuntime::initialize() {
  requireMode(MoEMode::LATENCY);
  latencyContext_->initialize();
}

void* MoERuntime::dispatchOutputBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .dispatchOutputBuffer_;
}

void MoERuntime::dispatch(const DispatchRequest& request) {
  requireMode(MoEMode::LATENCY);
  const auto* latencyRequest = std::get_if<LatencyDispatchRequest>(&request.value_);
  if (latencyRequest == nullptr) {
    throw std::invalid_argument("Throughput dispatch is not available in this build");
  }
  launchLatencyDispatch(*latencyRequest);
}

void MoERuntime::combine(const CombineRequest& request) {
  requireMode(MoEMode::LATENCY);
  const auto* latencyRequest = std::get_if<LatencyCombineRequest>(&request.value_);
  if (latencyRequest == nullptr) {
    throw std::invalid_argument("Throughput combine is not available in this build");
  }
  launchLatencyCombine(*latencyRequest);
}

std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, DispatchLayout outputLayout,
                                             CombineMode combineMode) {
  return std::make_shared<MoERuntime>(communicator, mode, maxTokensPerRank, hidden, numExperts, numTopk, outputLayout,
                                      combineMode);
}

}  // namespace ep
}  // namespace mscclpp
