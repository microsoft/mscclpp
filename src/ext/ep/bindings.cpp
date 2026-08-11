// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// nanobind module definition for the MSCCL++ EP extension.
//
// One `MoERuntime` class is exposed, with a torch-free, raw-pointer (uintptr_t)
// boundary so the module never links libtorch. `MoEMode` selects the backend at
// construction:
//   - MoEMode.LOW_LATENCY  -> `ll_*` methods (dispatch/combine).
//   - MoEMode.HIGH_THROUGHPUT -> `ht_*` methods. Dynamic recv sizing uses an
//     explicit multi-step API (ht_compute_dispatch_counts -> ht_notify_dispatch
//     -> caller allocates -> ht_dispatch).
// The two backends keep separate method prefixes because their call protocols
// genuinely differ; calling the other mode's methods raises.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstdint>
#include <stdexcept>

#include "api.cuh"
#include "config.hpp"
#include "high-throughput/config.cuh"
#include "ht_runtime.hpp"
#include "ll_runtime.hpp"
#include "moe_runtime.hpp"

namespace nb = nanobind;

namespace {

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

template <typename Runtime>
Runtime& narrow(mscclpp::ep::MoERuntime& runtime, const char* expectedMode) {
  auto* concrete = dynamic_cast<Runtime*>(&runtime);
  if (concrete == nullptr) {
    throw std::runtime_error(std::string("MoE runtime was not created with MoEMode::") + expectedMode);
  }
  return *concrete;
}

template <typename Runtime>
const Runtime& narrow(const mscclpp::ep::MoERuntime& runtime, const char* expectedMode) {
  auto* concrete = dynamic_cast<const Runtime*>(&runtime);
  if (concrete == nullptr) {
    throw std::runtime_error(std::string("MoE runtime was not created with MoEMode::") + expectedMode);
  }
  return *concrete;
}

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  nb::module_::import_("mscclpp._mscclpp");

  nb::enum_<mscclpp::ep::MoEMode>(m, "MoEMode")
      .value("LOW_LATENCY", mscclpp::ep::MoEMode::LOW_LATENCY)
      .value("HIGH_THROUGHPUT", mscclpp::ep::MoEMode::HIGH_THROUGHPUT);

  nb::enum_<mscclpp::ep::DispatchLayout>(m, "DispatchLayout")
      .value("EXPERT_MAJOR", mscclpp::ep::DispatchLayout::EXPERT_MAJOR)
      .value("TOKEN_MAJOR", mscclpp::ep::DispatchLayout::TOKEN_MAJOR)
      .value("RANK_MAJOR", mscclpp::ep::DispatchLayout::RANK_MAJOR);

  nb::enum_<mscclpp::ep::low_latency::CombineMode>(m, "CombineMode")
      .value("RANK_LOCAL_REDUCE", mscclpp::ep::low_latency::CombineMode::RANK_LOCAL_REDUCE)
      .value("DIRECT_SEND", mscclpp::ep::low_latency::CombineMode::DIRECT_SEND);
  nb::enum_<mscclpp::ep::low_latency::DispatchDataType>(m, "DispatchDataType")
      .value("BF16", mscclpp::ep::low_latency::DispatchDataType::BF16)
      .value("FP8_E4M3", mscclpp::ep::low_latency::DispatchDataType::FP8_E4M3);

  nb::class_<mscclpp::ep::high_throughput::Config>(m, "Config")
      .def(nb::init<int>(), nb::arg("num_sms") = 20)
      .def_ro("num_sms", &mscclpp::ep::high_throughput::Config::numSms_);

  m.def("create_moe_runtime", &mscclpp::ep::createMoERuntime, nb::arg("comm"), nb::arg("mode"),
        nb::arg("max_tokens_per_rank") = 0, nb::arg("hidden") = 0, nb::arg("num_experts") = 0, nb::arg("num_topk") = 0,
        nb::arg("max_hidden_bytes") = 0, nb::arg("num_sms") = 20,
        nb::arg("output_layout") = mscclpp::ep::DispatchLayout::EXPERT_MAJOR,
        "Create the MoE backend selected by mode; returns a shared MoERuntime handle.");

  nb::class_<mscclpp::ep::MoERuntime>(m, "MoERuntime")
      .def_prop_ro("mode", &mscclpp::ep::MoERuntime::mode)
      .def("is_available", &mscclpp::ep::MoERuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoERuntime::isInternodeAvailable)
      .def(
          "output_topk_ids_buffer_ptr",
          [](const mscclpp::ep::MoERuntime& self, int maxTokensPerRank) {
            return reinterpret_cast<uintptr_t>(narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                                                   .outputTopkIdsBuffer(maxTokensPerRank));
          },
          nb::arg("max_tokens_per_rank") = 0)
      .def(
          "output_topk_weights_buffer_ptr",
          [](const mscclpp::ep::MoERuntime& self, int maxTokensPerRank) {
            return reinterpret_cast<uintptr_t>(narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                                                   .outputTopkWeightsBuffer(maxTokensPerRank));
          },
          nb::arg("max_tokens_per_rank") = 0)
      .def(
          "output_tokens_buffer_ptr",
          [](const mscclpp::ep::MoERuntime& self, int maxTokensPerRank) {
            return reinterpret_cast<uintptr_t>(narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                                                   .outputTokensBuffer(maxTokensPerRank));
          },
          nb::arg("max_tokens_per_rank") = 0)
      .def(
          "expert_output_buffer_ptr",
          [](const mscclpp::ep::MoERuntime& self, int maxTokensPerRank) {
            return reinterpret_cast<uintptr_t>(narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                                                   .expertOutputBuffer(maxTokensPerRank));
          },
          nb::arg("max_tokens_per_rank") = 0)
      .def(
          "ll_dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t outputPtr, uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputTopkIdxPtr,
             uintptr_t outputTopkWeightsPtr, uintptr_t outputLayoutRangePtr, uintptr_t outputCountPtr, int numTokens,
             int hidden, int numTopk, int maxTokensPerRank, int numExperts, int invalidTokenExpertId,
             mscclpp::ep::DispatchLayout dispatchLayout, mscclpp::ep::low_latency::DispatchDataType dispatchDataType,
             int numBlocks, uintptr_t streamPtr) {
            narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                .dispatch(
                    ptr(outputPtr), ptr(outputScalesPtr), reinterpret_cast<int*>(ptr(outputSrcInfoPtr)),
                    reinterpret_cast<int*>(ptr(outputTopkIdxPtr)), reinterpret_cast<float*>(ptr(outputTopkWeightsPtr)),
                    reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)), reinterpret_cast<int*>(ptr(outputCountPtr)),
                    ptr(inputPtr), reinterpret_cast<int64_t*>(ptr(topkIdxPtr)),
                    reinterpret_cast<float*>(ptr(topkWeightsPtr)), numTokens, hidden, numTopk, maxTokensPerRank,
                    numExperts, invalidTokenExpertId, dispatchLayout, dispatchDataType, numBlocks, stream(streamPtr));
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("output_ptr"),
          nb::arg("output_scales_ptr"), nb::arg("output_src_info_ptr"), nb::arg("output_topk_idx_ptr"),
          nb::arg("output_topk_weights_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("max_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("invalid_token_expert_id"), nb::arg("dispatch_layout"),
          nb::arg("dispatch_data_type"), nb::arg("num_blocks"), nb::arg("stream_ptr"))
      .def(
          "ll_combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t srcInfoPtr, uintptr_t layoutRangePtr, uintptr_t outputPtr, int numTokens, int hidden,
             int numTopk, int maxTokensPerRank, int numExperts, mscclpp::ep::DispatchLayout dispatchLayout,
             mscclpp::ep::low_latency::DispatchDataType dispatchDataType, mscclpp::ep::low_latency::CombineMode mode,
             int numBlocks, uintptr_t streamPtr) {
            narrow<mscclpp::ep::MoELowLatencyRuntime>(self, "LOW_LATENCY")
                .combine(ptr(outputPtr), ptr(expertOutputPtr), reinterpret_cast<int64_t*>(ptr(topkIdxPtr)),
                         reinterpret_cast<float*>(ptr(topkWeightsPtr)), reinterpret_cast<int*>(ptr(srcInfoPtr)),
                         reinterpret_cast<int64_t*>(ptr(layoutRangePtr)), numTokens, hidden, numTopk, maxTokensPerRank,
                         numExperts, dispatchLayout, dispatchDataType, mode, numBlocks, stream(streamPtr));
          },
          nb::arg("expert_output_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("src_info_ptr"),
          nb::arg("layout_range_ptr"), nb::arg("output_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("num_topk"), nb::arg("max_tokens_per_rank"), nb::arg("num_experts"), nb::arg("dispatch_layout"),
          nb::arg("dispatch_data_type"), nb::arg("mode"), nb::arg("num_blocks"), nb::arg("stream_ptr"))
      .def(
          "ht_compute_dispatch_counts",
          [](mscclpp::ep::MoERuntime& self, uintptr_t num_tokens_per_rank_ptr, uintptr_t num_tokens_per_expert_ptr,
             uintptr_t is_token_in_rank_ptr, uintptr_t topk_idx_ptr, int num_tokens, int num_topk, int num_experts,
             uintptr_t stream_ptr) {
            narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                .computeDispatchCounts(reinterpret_cast<int*>(ptr(num_tokens_per_rank_ptr)),
                                       reinterpret_cast<int*>(ptr(num_tokens_per_expert_ptr)),
                                       reinterpret_cast<bool*>(ptr(is_token_in_rank_ptr)),
                                       reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)), num_tokens, num_topk,
                                       num_experts, stream(stream_ptr));
          },
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("num_tokens"), nb::arg("num_topk"), nb::arg("num_experts"),
          nb::arg("stream_ptr"))
      .def("ht_get_dispatch_num_channels",
           [](const mscclpp::ep::MoERuntime& self, int x_element_size) {
             return narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                 .getDispatchNumChannels(x_element_size);
           })
      .def("ht_resolve_recv_x_buffer",
           [](const mscclpp::ep::MoERuntime& self, int num_tokens, int num_recv_tokens, int hidden,
              int x_element_size) -> uintptr_t {
             return reinterpret_cast<uintptr_t>(
                 narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                     .resolveRecvXBuffer(num_tokens, num_recv_tokens, hidden, x_element_size));
           })
      .def(
          "ht_notify_dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr,
             uintptr_t num_recv_tokens_per_expert_ptr, uintptr_t num_tokens_per_rank_ptr,
             uintptr_t num_tokens_per_expert_ptr, uintptr_t is_token_in_rank_ptr, int num_tokens, int num_experts,
             int x_element_size, int expert_alignment, uintptr_t stream_ptr) {
            return narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                .notifyDispatch(reinterpret_cast<int*>(ptr(rank_prefix_matrix_ptr)),
                                reinterpret_cast<int*>(ptr(channel_prefix_matrix_ptr)),
                                reinterpret_cast<int*>(ptr(num_recv_tokens_per_expert_ptr)),
                                reinterpret_cast<const int*>(ptr(num_tokens_per_rank_ptr)),
                                reinterpret_cast<const int*>(ptr(num_tokens_per_expert_ptr)),
                                reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)), num_tokens, num_experts,
                                x_element_size, expert_alignment, stream(stream_ptr));
          },
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"),
          nb::arg("num_recv_tokens_per_expert_ptr"), nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("x_element_size"), nb::arg("expert_alignment"), nb::arg("stream_ptr"))
      .def(
          "ht_dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t recv_x_ptr, uintptr_t recv_x_scales_ptr,
             uintptr_t recv_topk_idx_ptr, uintptr_t recv_topk_weights_ptr, uintptr_t send_head_ptr, uintptr_t x_ptr,
             uintptr_t x_scales_ptr, uintptr_t topk_idx_ptr, uintptr_t topk_weights_ptr, uintptr_t is_token_in_rank_ptr,
             uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr, int num_tokens, int hidden,
             int num_topk, int num_scales, int num_experts, int x_element_size, int num_recv_tokens, bool cached_mode,
             uintptr_t stream_ptr) {
            narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                .dispatch(ptr(recv_x_ptr), reinterpret_cast<float*>(ptr(recv_x_scales_ptr)),
                          reinterpret_cast<int64_t*>(ptr(recv_topk_idx_ptr)),
                          reinterpret_cast<float*>(ptr(recv_topk_weights_ptr)),
                          reinterpret_cast<int*>(ptr(send_head_ptr)), ptr(x_ptr),
                          reinterpret_cast<const float*>(ptr(x_scales_ptr)),
                          reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)),
                          reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                          reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)),
                          reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                          reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)), num_tokens, hidden, num_topk,
                          num_scales, num_experts, x_element_size, num_recv_tokens, cached_mode, stream(stream_ptr));
          },
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"), nb::arg("send_head_ptr"), nb::arg("x_ptr"), nb::arg("x_scales_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"), nb::arg("num_tokens"),
          nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_scales"), nb::arg("num_experts"),
          nb::arg("x_element_size"), nb::arg("num_recv_tokens"), nb::arg("cached_mode"), nb::arg("stream_ptr"))
      .def(
          "ht_combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t combined_x_ptr, uintptr_t combined_topk_weights_ptr,
             uintptr_t x_ptr, uintptr_t topk_weights_ptr, uintptr_t send_head_ptr, int num_input_tokens,
             int num_output_tokens, int hidden, int num_topk, int x_element_size, uintptr_t stream_ptr) {
            narrow<mscclpp::ep::MoEHighThroughputRuntime>(self, "HIGH_THROUGHPUT")
                .combine(ptr(combined_x_ptr), reinterpret_cast<float*>(ptr(combined_topk_weights_ptr)), ptr(x_ptr),
                         reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                         reinterpret_cast<const int*>(ptr(send_head_ptr)), num_input_tokens, num_output_tokens, hidden,
                         num_topk, x_element_size, stream(stream_ptr));
          },
          nb::arg("combined_x_ptr"), nb::arg("combined_topk_weights_ptr"), nb::arg("x_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("send_head_ptr"), nb::arg("num_input_tokens"),
          nb::arg("num_output_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("x_element_size"),
          nb::arg("stream_ptr"));
}
