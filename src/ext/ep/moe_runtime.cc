// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include "ht_runtime.hpp"
#include "ll_runtime.hpp"

namespace mscclpp {
namespace ep {

std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numSms, DispatchLayout outputLayout) {
  if (mode == MoEMode::LOW_LATENCY) {
    return std::make_shared<MoELowLatencyRuntime>(communicator, maxTokensPerRank, hidden, numExperts, numTopk,
                                                  outputLayout);
  }
  return std::make_shared<MoEHighThroughputRuntime>(communicator, maxHiddenBytes, high_throughput::Config(numSms));
}

}  // namespace ep
}  // namespace mscclpp
