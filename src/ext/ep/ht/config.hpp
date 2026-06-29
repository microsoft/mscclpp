// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "../config.hpp"
#include "../kernels/api.cuh"
#include <cstdlib>

namespace mscclpp {
namespace ep {

// inc7: resolve the flat all-sender dispatch channel count (== block count).
// Default num_sms/2; MSCCLPP_EP_DISPATCH_NSM (flat path only) overrides it, clamped
// to [1, num_sms]. The flat path launches one sender block per channel (no forwarder),
// so it can use the FULL SM budget with a SINGLE knob. Non-flat/unset -> num_sms/2.
inline int ep_flat_dispatch_channels(int num_sms) {
  const char* f = std::getenv("MSCCLPP_EP_FLAT");
  const char* d = std::getenv("MSCCLPP_EP_DIRECT");
  if (!(f != nullptr && std::atoi(f) != 0 && d != nullptr && std::atoi(d) != 0)) return num_sms / 2;
  const char* e = std::getenv("MSCCLPP_EP_DISPATCH_NSM");
  if (e == nullptr) return num_sms / 2;
  const int n = std::atoi(e);
  if (n < 1) return num_sms / 2;
  return n < num_sms ? n : num_sms;
}

// Channel count the RDMA/NVL buffers must hold: max of the flat dispatch channels
// and the default num_sms/2 (combine grid). Keeps per-channel offsets in bounds when
// DISPATCH_NSM raises/lowers the dispatch channel count relative to num_sms/2.
inline int ep_buffer_channels(int num_sms) {
  const int dc = ep_flat_dispatch_channels(num_sms);
  const int half = num_sms / 2;
  return dc > half ? dc : half;
}

struct Config {
  int num_sms;
  int num_max_nvl_chunked_send_tokens;
  int num_max_nvl_chunked_recv_tokens;
  int num_max_rdma_chunked_send_tokens;
  int num_max_rdma_chunked_recv_tokens;

  Config(int num_sms, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
         int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
    EP_HOST_ASSERT(num_sms >= 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

    // Ceil up RDMA buffer size
    this->num_max_rdma_chunked_recv_tokens =
        align<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
    // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
  }

  size_t get_nvl_base_bytes(size_t hidden_bytes, int num_ranks) const {
    // Below are some assumptions
    // TODO: add assertions
    constexpr int kNumMaxTopK = 128;
    constexpr int kNumMaxScales = 128;
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
    const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
    const auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
    const int num_channels = ep_buffer_channels(num_sms);

    size_t num_bytes = 0;
    num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(int64_t);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
    num_bytes = ((num_bytes + 127) / 128) * 128;
    return num_bytes;
  }

#ifdef EP_DISPATCH_NCCLEP
  static constexpr int kEpRecvPoolMaxTokens = 65536;
  static constexpr int64_t kEpRecvPoolMaxHiddenBytes = 16384;
  size_t get_recv_pool_header_bytes(int num_ranks) const {
    return ((static_cast<size_t>(num_ranks) * sizeof(int) + 127) / 128) * 128;
  }
  static size_t recv_pool_bytes_static(int num_ranks) {
    size_t header = ((static_cast<size_t>(num_ranks) * sizeof(int) + 127) / 128) * 128;
    size_t b = header + static_cast<size_t>(kEpRecvPoolMaxTokens) * static_cast<size_t>(kEpRecvPoolMaxHiddenBytes);
    return ((b + 127) / 128) * 128;
  }
#endif

  size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    return get_nvl_base_bytes(hidden_bytes, num_ranks);
  }

  size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
    // Legacy mode
    if (num_ranks <= NUM_MAX_NVL_PEERS) return 0;

    // Below are some assumptions
    // TODO: add assertions
    constexpr int kNumMaxTopK = 128;
    constexpr int kNumMaxScales = 128;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_sms % 2 == 0);
    const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int num_channels = ep_buffer_channels(num_sms);

    size_t num_bytes = 0;
    num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
    num_bytes +=
        num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(int64_t) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
    num_bytes += num_channels * num_rdma_ranks * sizeof(uint64_t) * 2;
    num_bytes = ((num_bytes + 127) / 128) * 128;
    return num_bytes;
  }
};

}  // namespace ep
}  // namespace mscclpp
