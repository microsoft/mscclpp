// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>

#include "../config.hpp"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {

struct Config {
  static constexpr int MaxTopk = 32;
  static constexpr int MaxScales = 128;
  static constexpr int RecvPoolMaxTokens = 65536;
  static constexpr int64_t RecvPoolMaxHiddenBytes = 16384;
  static constexpr int64_t RecvPoolMetaBytes =
      ((MaxTopk * (sizeof(int) + sizeof(float)) + MaxScales * sizeof(float) + NUM_BUFFER_ALIGNMENT_BYTES - 1) /
       NUM_BUFFER_ALIGNMENT_BYTES) *
      NUM_BUFFER_ALIGNMENT_BYTES;

  int numSms_;

  explicit Config(int numSms) : numSms_(numSms) { EP_HOST_ASSERT(numSms > 0); }

  size_t controlBufferBytes(int numRanks) const {
    EP_HOST_ASSERT(numRanks == 2 || numRanks == 4 || numRanks == 8 || numRanks == 16);

    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t prefixBytes = ranks * ranks * sizeof(int);
    const size_t expertScratchBytes = ranks * NUM_MAX_LOCAL_EXPERTS * sizeof(int);
    return configAlign<size_t>(prefixBytes + expertScratchBytes, NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t recvPoolHeaderBytes(int numRanks) {
    return configAlign<size_t>(static_cast<size_t>(numRanks) * sizeof(int), NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t recvPoolMetadataOffset(int numRanks) {
    const size_t hiddenBytes = static_cast<size_t>(RecvPoolMaxTokens) * static_cast<size_t>(RecvPoolMaxHiddenBytes);
    return configAlign<size_t>(recvPoolHeaderBytes(numRanks) + hiddenBytes, NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t recvPoolHiddenBytes(int numRanks) {
    return recvPoolMetadataOffset(numRanks) - recvPoolHeaderBytes(numRanks);
  }

  static size_t recvPoolBytes(int numRanks) {
    const size_t bytes = recvPoolMetadataOffset(numRanks) + static_cast<size_t>(RecvPoolMaxTokens) * RecvPoolMetaBytes;
    return configAlign<size_t>(bytes, NUM_BUFFER_ALIGNMENT_BYTES);
  }
};

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
