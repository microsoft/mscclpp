// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../config.hpp"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

struct Config {
  static constexpr int MaxTopk = 32;
  static constexpr int MaxScales = 128;
  static constexpr int RecvPoolMaxTokens = 65536;
  static constexpr int64_t RecvPoolMaxHiddenBytes = 16384;
  static constexpr int64_t RecvPoolMetaBytes = 128;

  int num_sms;
  int num_max_nvl_chunked_send_tokens;
  int num_max_nvl_chunked_recv_tokens;

  Config(int num_sms, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens) {
    EP_HOST_ASSERT(num_sms > 0 and num_sms % 2 == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
  }

  size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    EP_HOST_ASSERT(hidden_bytes > 0);
    EP_HOST_ASSERT(num_ranks == 2 or num_ranks == 4 or num_ranks == 8 or num_ranks == 16);

    const size_t ranks = static_cast<size_t>(num_ranks);
    const size_t recv_tokens = static_cast<size_t>(num_max_nvl_chunked_recv_tokens);
    const size_t ring_channels = static_cast<size_t>(num_sms / 2);
    const size_t all_sender_channels = static_cast<size_t>(num_sms);
    const size_t prefix_bytes = ranks * ranks * sizeof(int);
    const size_t expert_scratch_bytes = ranks * NUM_MAX_LOCAL_EXPERTS * sizeof(int);

    const size_t dispatch_metadata_bytes = 4 * ring_channels * ranks * sizeof(int);
    const size_t dispatch_token_bytes =
        hidden_bytes + sizeof(int) + MaxTopk * sizeof(int64_t) + MaxTopk * sizeof(float) + MaxScales * sizeof(float);
    const size_t dispatch_bytes =
        prefix_bytes + std::max(expert_scratch_bytes,
                                dispatch_metadata_bytes + ring_channels * ranks * recv_tokens * dispatch_token_bytes);

    const size_t combine_metadata_bytes = 2 * ring_channels * ranks * sizeof(int);
    const size_t combine_token_bytes = hidden_bytes + sizeof(int) + MaxTopk * sizeof(float);
    const size_t combine_bytes = combine_metadata_bytes + ring_channels * ranks * recv_tokens * combine_token_bytes;

    const size_t all_sender_bytes =
        prefix_bytes + std::max(expert_scratch_bytes, 4 * all_sender_channels * ranks * sizeof(int));
    return configAlign<size_t>(std::max({dispatch_bytes, combine_bytes, all_sender_bytes}), NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t get_recv_pool_header_bytes(int num_ranks) {
    return configAlign<size_t>(static_cast<size_t>(num_ranks) * sizeof(int), NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t get_recv_pool_meta_base(int num_ranks) {
    const size_t hidden_bytes = static_cast<size_t>(RecvPoolMaxTokens) * static_cast<size_t>(RecvPoolMaxHiddenBytes);
    return configAlign<size_t>(get_recv_pool_header_bytes(num_ranks) + hidden_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
  }

  static size_t get_recv_pool_hidden_bytes(int num_ranks) {
    return get_recv_pool_meta_base(num_ranks) - get_recv_pool_header_bytes(num_ranks);
  }

  static size_t recv_pool_bytes_static(int num_ranks) {
    const size_t bytes =
        get_recv_pool_meta_base(num_ranks) + static_cast<size_t>(RecvPoolMaxTokens) * RecvPoolMetaBytes;
    return configAlign<size_t>(bytes, NUM_BUFFER_ALIGNMENT_BYTES);
  }
};

}  // namespace ep
}  // namespace mscclpp
