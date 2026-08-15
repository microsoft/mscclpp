// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>

#include "../config.hpp"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

struct RecvPoolConfig {
  static constexpr int MaxTopk = 32;
  static constexpr int MaxScales = 128;
  static constexpr int MaxLocalExperts = 1024;
  // Dispatch receive sizes are data-dependent, so peers write into this fixed,
  // setup-time mapped internal pool before Python can expose the exact-size view.
  static constexpr int RecvPoolMaxTokens = 65536;
  static constexpr int64_t RecvPoolMaxHiddenBytes = 16384;
  static constexpr int64_t RecvPoolMetaBytes =
      ((MaxTopk * (sizeof(int) + sizeof(float)) + MaxScales * sizeof(float) + BufferAlignmentBytes - 1) /
       BufferAlignmentBytes) *
      BufferAlignmentBytes;

  int numBlocks_;

  explicit RecvPoolConfig(int numBlocks) : numBlocks_(numBlocks) { EP_HOST_ASSERT(numBlocks > 0); }

  size_t controlBufferBytes(int numRanks) const {
    EP_HOST_ASSERT(numRanks == 2 || numRanks == 4 || numRanks == 8 || numRanks == 16);

    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t prefixBytes = ranks * ranks * sizeof(int);
    const size_t expertScratchBytes = ranks * MaxLocalExperts * sizeof(int);
    return configAlign<size_t>(prefixBytes + expertScratchBytes, BufferAlignmentBytes);
  }

  static size_t recvPoolHeaderBytes(int numRanks) {
    return configAlign<size_t>(static_cast<size_t>(numRanks) * sizeof(int), BufferAlignmentBytes);
  }

  static size_t recvPoolMetadataOffset(int numRanks) {
    const size_t hiddenBytes = static_cast<size_t>(RecvPoolMaxTokens) * static_cast<size_t>(RecvPoolMaxHiddenBytes);
    return configAlign<size_t>(recvPoolHeaderBytes(numRanks) + hiddenBytes, BufferAlignmentBytes);
  }

  static size_t recvPoolHiddenBytes(int numRanks) {
    return recvPoolMetadataOffset(numRanks) - recvPoolHeaderBytes(numRanks);
  }

  static size_t recvPoolBytes(int numRanks) {
    const size_t bytes = recvPoolMetadataOffset(numRanks) + static_cast<size_t>(RecvPoolMaxTokens) * RecvPoolMetaBytes;
    return configAlign<size_t>(bytes, BufferAlignmentBytes);
  }
};

}  // namespace ep
}  // namespace mscclpp
