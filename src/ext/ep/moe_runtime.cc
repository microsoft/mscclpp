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
                       int numExperts, int numTopk, int64_t maxHiddenBytes, int numBlocks, DispatchLayout outputLayout,
                       CombineMode combineMode, int etpSize, EtpRankOrder etpRankOrder, EtpReduceMode etpReduceMode,
                       EtpDispatchMode etpDispatchMode)
    : bootstrap_(communicator.bootstrap()),
      mode_(mode),
      rank_(bootstrap_->getRank()),
      numRanks_(bootstrap_->getNranks()),
      numNvlRanks_(std::min(numRanks_, bootstrap_->getNranksPerNode())),
      numRanksPerIpcDomain_(std::max(numNvlRanks_, std::min(numRanks_, bootstrap_->getNranksPerIpcDomain()))),
      topology_(numRanks_, etpSize, etpRankOrder) {
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
  EP_HOST_ASSERT(numNvlRanks_ > 0);
  EP_HOST_ASSERT(etpSize > 0 && numRanks_ % etpSize == 0);
  EP_HOST_ASSERT(topology_.epSize * topology_.etpSize == numRanks_);
  if (topology_.isEtpEnabled()) {
    // ETP requires every peer of an EP group to be directly mappable.
    EP_HOST_ASSERT(numRanksPerIpcDomain_ >= numRanks_);
    EP_HOST_ASSERT(etpReduceMode != EtpReduceMode::GROUP_NVLS);
  }

  switch (mode_) {
    case MoEMode::LATENCY:
      latencyContext_ = std::make_unique<LatencyContext>(
          communicator, rank_, numRanks_, numNvlRanks_, numRanksPerIpcDomain_, maxTokensPerRank, hidden, numExperts,
          numTopk, outputLayout, combineMode, topology_, etpReduceMode, etpDispatchMode);
      available_ = latencyContext_->available_;
      break;
    case MoEMode::THROUGHPUT:
      EP_HOST_ASSERT(!topology_.isEtpEnabled() && "THROUGHPUT mode does not support etpSize > 1 yet");
      throughputContext_ = std::make_unique<ThroughputContext>(communicator, rank_, numRanks_, numNvlRanks_,
                                                               numRanksPerIpcDomain_, maxTokensPerRank, maxHiddenBytes,
                                                               outputLayout, RecvPoolConfig(numBlocks), topology_);
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

void MoERuntime::initialize() {
  switch (mode_) {
    case MoEMode::LATENCY:
      latencyContext_->initialize();
      return;
    case MoEMode::THROUGHPUT:
      throughputContext_->initialize();
      return;
    default:
      throw std::invalid_argument("Unsupported MoE runtime mode");
  }
}

void* MoERuntime::dispatchOutputBuffer() const {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto& context = *latencyContext_;
      EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
      return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_,
                                  context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                                  context.combineMode_, context.topology_, context.etpReduceMode_,
                                  context.etpDispatchMode_)
          .dispatchOutputBuffer_;
    }
    case MoEMode::THROUGHPUT: {
      const auto& context = *throughputContext_;
      EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
      if (!context.collectiveDirectReady_) return nullptr;
      return static_cast<uint8_t*>(context.recvPoolPtrs_[context.rank_]) +
             RecvPoolConfig::recvPoolHeaderBytes(context.numRanks_);
    }
    default:
      throw std::invalid_argument("Unsupported MoE runtime mode");
  }
}

void MoERuntime::dispatch(const DispatchRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<LatencyDispatchRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        throw std::invalid_argument("Latency runtime requires a latency dispatch request");
      }
      launchLatencyDispatch(*latencyRequest);
      return;
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<ThroughputDispatchRequest>(&request.value_);
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

void MoERuntime::combine(const CombineRequest& request) {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto* latencyRequest = std::get_if<LatencyCombineRequest>(&request.value_);
      if (latencyRequest == nullptr) {
        throw std::invalid_argument("Latency runtime requires a latency combine request");
      }
      launchLatencyCombine(*latencyRequest);
      return;
    }
    case MoEMode::THROUGHPUT: {
      const auto* throughputRequest = std::get_if<ThroughputCombineRequest>(&request.value_);
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
                                             int numBlocks, DispatchLayout outputLayout, CombineMode combineMode,
                                             int etpSize, EtpRankOrder etpRankOrder, EtpReduceMode etpReduceMode,
                                             EtpDispatchMode etpDispatchMode) {
  return std::make_shared<MoERuntime>(communicator, mode, maxTokensPerRank, hidden, numExperts, numTopk, maxHiddenBytes,
                                      numBlocks, outputLayout, combineMode, etpSize, etpRankOrder, etpReduceMode,
                                      etpDispatchMode);
}

}  // namespace ep
}  // namespace mscclpp
