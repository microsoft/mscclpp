// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>

#include "runtime_base.hpp"

namespace mscclpp {
namespace ep {

/// Create the MoE backend selected by @p mode.
///
/// The result is a `MoERuntime`, so callers hold one handle regardless of
/// backend and query `mode()` to tell them apart.
///
/// @param communicator Communicator shared with the rest of MSCCL++.
/// @param mode Backend to construct.
/// @param maxTokensPerRank Low-latency token capacity per rank.
/// @param hidden Low-latency hidden size.
/// @param numExperts Low-latency total expert count.
/// @param numTopk Low-latency routed experts per token.
/// @param maxHiddenBytes High-throughput maximum hidden-size bytes.
/// @param numSms High-throughput SM budget for the comms kernels.
/// @return The constructed backend.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numSms);

}  // namespace ep
}  // namespace mscclpp
