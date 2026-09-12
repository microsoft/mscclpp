// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/ext/ep/moe_runtime.hpp>

#include "exception.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

const PrepareHandle::Impl& PrepareHandle::checked() const {
  if (!impl_) {
    EP_THROW("Invalid or empty preparation handle");
  }
  const auto owner = impl_->owner_.lock();
  if (!owner) {
    EP_THROW("Expired preparation handle");
  }
  if (impl_->epoch_ != owner->prepareEpoch_) {
    EP_THROW("Stale preparation handle: a newer preparation has replaced its metadata");
  }
  return *impl_;
}

const int* PrepareHandle::numRecvTokensDevice() const { return checked().numRecvTokens_; }

const int* PrepareHandle::outputCountsDevice() const { return checked().outputCounts_; }

int PrepareHandle::numOutputCounts() const { return checked().numOutputCounts_; }

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

  switch (mode_) {
    case MoEMode::LATENCY:
      latencyContext_ = std::make_shared<LatencyRuntimeContext>(communicator, rank_, numRanks_, numNvlRanks_,
                                                                numRanksPerIpcDomain_, maxTokensPerRank, hidden,
                                                                numExperts, numTopk, outputLayout, combineMode);
      available_ = latencyContext_->available_;
      break;
    case MoEMode::THROUGHPUT:
      throughputContext_ =
          std::make_shared<ThroughputRuntimeContext>(communicator, rank_, numRanks_, numRanksPerIpcDomain_,
                                                     maxTokensPerRank, hidden, numExperts, numTopk, outputLayout);
      available_ = throughputContext_->available_;
      break;
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

MoERuntime::~MoERuntime() noexcept(false) = default;

void MoERuntime::requireMode(MoEMode expected) const {
  if (mode_ != expected) {
    EP_THROW(expected == MoEMode::LATENCY ? "MoE runtime was not created with MoEMode::LATENCY"
                                          : "MoE runtime was not created with MoEMode::THROUGHPUT");
  }
}

void MoERuntime::initialize() {
  switch (mode_) {
    case MoEMode::LATENCY:
      latencyContext_->initialize();
      return;
    case MoEMode::THROUGHPUT:
      throughputContext_->initialize();
      return;
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

void* MoERuntime::dispatchOutputBuffer() const {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto& context = *latencyContext_;
      EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
      return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_,
                                  context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                                  context.combineMode_)
          .dispatchOutputBuffer_;
    }
    case MoEMode::THROUGHPUT: {
      const auto& context = *throughputContext_;
      EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
      return ThroughputStorageLayout(context.symmetricBuffer_, context.numRanks_).recvBuffer_;
    }
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

DispatchHandle MoERuntime::dispatch(const DispatchRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<LatencyDispatchRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        EP_THROW("Latency runtime requires a latency dispatch request");
      }
      return launchLatencyDispatch(*latencyRequest);
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<ThroughputDispatchRequest>(&request.value_);
      if (throughputRequest == nullptr) {
        EP_THROW("Throughput runtime requires a throughput dispatch request");
      }
      return launchThroughputDispatch(*throughputRequest);
    }
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

void MoERuntime::combine(const CombineRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<LatencyCombineRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        EP_THROW("Latency runtime requires a latency combine request");
      }
      launchLatencyCombine(*latencyRequest);
      return;
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<ThroughputCombineRequest>(&request.value_);
      if (throughputRequest == nullptr) {
        EP_THROW("Throughput runtime requires a throughput combine request");
      }
      launchThroughputCombine(*throughputRequest);
      return;
    }
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, DispatchLayout outputLayout,
                                             CombineMode combineMode) {
  return std::make_shared<MoERuntime>(communicator, mode, maxTokensPerRank, hidden, numExperts, numTopk, outputLayout,
                                      combineMode);
}

}  // namespace ep
}  // namespace mscclpp
