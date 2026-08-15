// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <algorithm>
#include <stdexcept>

#include "exception.cuh"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden,
                       int numExperts, int numTopk, int64_t maxHiddenBytes, int numBlocks, DispatchLayout outputLayout)
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
      latencyContext_ =
          std::make_unique<detail::LatencyContext>(communicator, rank_, numRanks_, numNvlRanks_, numRanksPerIpcDomain_,
                                                   maxTokensPerRank, hidden, numExperts, numTopk, outputLayout);
      available_ = latencyContext_->available_;
      break;
    case MoEMode::THROUGHPUT:
      throughputContext_ =
          std::make_unique<detail::ThroughputContext>(communicator, rank_, numRanks_, numNvlRanks_,
                                                      numRanksPerIpcDomain_, maxHiddenBytes, RecvPoolConfig(numBlocks));
      available_ = throughputContext_->available_;
      break;
    default:
      throw std::invalid_argument("Unsupported MoE runtime mode");
  }
}

MoERuntime::~MoERuntime() noexcept(false) = default;

void MoERuntime::requireMode(MoEMode expected) const {
  if (mode_ != expected) {
    throw std::runtime_error(expected == MoEMode::LATENCY ? "MoE runtime was not created with MoEMode::LATENCY"
                                                          : "MoE runtime was not created with MoEMode::THROUGHPUT");
  }
}

void MoERuntime::dispatch(const detail::DispatchRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<detail::LatencyDispatchRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        throw std::invalid_argument("Latency runtime requires a latency dispatch request");
      }
      launchLatencyDispatch(*latencyRequest);
      return;
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<detail::ThroughputDispatchRequest>(&request.value_);
      if (throughputRequest == nullptr) {
        throw std::invalid_argument("Throughput runtime requires a throughput dispatch request");
      }
      launchThroughputDispatch(*throughputRequest);
      return;
    }
    default:
      throw std::invalid_argument("Unsupported MoE runtime mode");
  }
}

void MoERuntime::combine(const detail::CombineRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<detail::LatencyCombineRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        throw std::invalid_argument("Latency runtime requires a latency combine request");
      }
      launchLatencyCombine(*latencyRequest);
      return;
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<detail::ThroughputCombineRequest>(&request.value_);
      if (throughputRequest == nullptr) {
        throw std::invalid_argument("Throughput runtime requires a throughput combine request");
      }
      launchThroughputCombine(*throughputRequest);
      return;
    }
    default:
      throw std::invalid_argument("Unsupported MoE runtime mode");
  }
}

std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numBlocks, DispatchLayout outputLayout) {
  return std::make_shared<MoERuntime>(communicator, mode, maxTokensPerRank, hidden, numExperts, numTopk, maxHiddenBytes,
                                      numBlocks, outputLayout);
}

}  // namespace ep
}  // namespace mscclpp
