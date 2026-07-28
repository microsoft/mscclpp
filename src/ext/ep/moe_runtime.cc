// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <stdexcept>

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden,
                       int numExperts, int numTopk, int64_t maxHiddenBytes, int numSms)
    : mode_(mode) {
  if (mode_ == MoEMode::LOW_LATENCY) {
    lowLatency_ = std::make_unique<MoELowLatencyRuntime>(communicator, maxTokensPerRank, hidden, numExperts, numTopk);
  } else {
    highThroughput_ =
        std::make_unique<MoEHighThroughputRuntime>(communicator, maxHiddenBytes, high_throughput::Config(numSms));
  }
}

MoERuntime::~MoERuntime() = default;

bool MoERuntime::isAvailable() const {
  return mode_ == MoEMode::LOW_LATENCY ? lowLatency_->isAvailable() : highThroughput_->isAvailable();
}

bool MoERuntime::isInternodeAvailable() const {
  return mode_ == MoEMode::LOW_LATENCY ? lowLatency_->isInternodeAvailable() : highThroughput_->isInternodeAvailable();
}

MoELowLatencyRuntime& MoERuntime::lowLatency() {
  if (lowLatency_ == nullptr) throw std::runtime_error("MoERuntime was not created with MoEMode::LOW_LATENCY");
  return *lowLatency_;
}

const MoELowLatencyRuntime& MoERuntime::lowLatency() const {
  if (lowLatency_ == nullptr) throw std::runtime_error("MoERuntime was not created with MoEMode::LOW_LATENCY");
  return *lowLatency_;
}

MoEHighThroughputRuntime& MoERuntime::highThroughput() {
  if (highThroughput_ == nullptr) throw std::runtime_error("MoERuntime was not created with MoEMode::HIGH_THROUGHPUT");
  return *highThroughput_;
}

const MoEHighThroughputRuntime& MoERuntime::highThroughput() const {
  if (highThroughput_ == nullptr) throw std::runtime_error("MoERuntime was not created with MoEMode::HIGH_THROUGHPUT");
  return *highThroughput_;
}

}  // namespace ep
}  // namespace mscclpp
