// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/ext/ep/moe_runtime.hpp>

#include "exception.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden,
                       int numExperts, int numTopk, DispatchLayout outputLayout, CombineMode combineMode)
    : communicator_(communicator),
      mode_(mode),
      rank_(communicator_.bootstrap()->getRank()),
      numRanks_(communicator_.bootstrap()->getNranks()),
      numRanksPerIpcDomain_(std::min(numRanks_, communicator_.bootstrap()->getNranksPerIpcDomain())) {
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
  EP_HOST_ASSERT(numRanksPerIpcDomain_ > 0);

  switch (mode_) {
    case MoEMode::LATENCY:
      latencyContext_ = std::make_shared<LatencyRuntimeContext>(communicator_, rank_, numRanks_, numRanksPerIpcDomain_,
                                                                maxTokensPerRank, hidden, numExperts, numTopk,
                                                                outputLayout, combineMode);
      available_ = latencyContext_->available_;
      break;
    case MoEMode::THROUGHPUT:
      throughputContext_ =
          std::make_shared<ThroughputRuntimeContext>(communicator_, rank_, numRanks_, numRanksPerIpcDomain_,
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

void* MoERuntime::outputTopkIdsBuffer() const {
  if (mode_ == MoEMode::THROUGHPUT) {
    const auto& context = *throughputContext_;
    EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
    return context.storageLayout().outputTopkIdsBuffer_;
  }
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_.get(), context.maxTokensPerRank_, context.hidden_,
                              context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                              context.combineMode_)
      .rankMajorTopkIdsBuffer_;
}

void* MoERuntime::outputTopkWeightsBuffer() const {
  if (mode_ == MoEMode::THROUGHPUT) {
    const auto& context = *throughputContext_;
    EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
    return context.storageLayout().outputTopkWeightsBuffer_;
  }
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_.get(), context.maxTokensPerRank_, context.hidden_,
                              context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                              context.combineMode_)
      .rankMajorTopkWeightsBuffer_;
}

void* MoERuntime::outputScalesBuffer() const {
  requireMode(MoEMode::THROUGHPUT);
  const auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  return context.storageLayout().outputScalesBuffer_;
}

void* MoERuntime::dispatchOutputBuffer() const {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto& context = *latencyContext_;
      EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
      return LatencyStorageLayout(context.symmetricBuffer_.get(), context.maxTokensPerRank_, context.hidden_,
                                  context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                                  context.combineMode_)
          .dispatchOutputBuffer_;
    }
    case MoEMode::THROUGHPUT: {
      const auto& context = *throughputContext_;
      EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
      return context.storageLayout().recvBuffer_;
    }
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

void* MoERuntime::combineInputBuffer() const {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto& context = *latencyContext_;
      EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
      EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
      return LatencyStorageLayout(context.symmetricBuffer_.get(), context.maxTokensPerRank_, context.hidden_,
                                  context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                                  context.combineMode_)
          .combineBuffer_;
    }
    case MoEMode::THROUGHPUT: {
      const auto& context = *throughputContext_;
      EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
      return context.storageLayout().combineBuffer_;
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
