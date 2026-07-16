// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// nanobind module definition for the MSCCL++ EP extension.
//
// Two backends are exposed in one module, both with a torch-free, raw-pointer
// (uintptr_t) boundary so the module never links libtorch:
//   - MoERuntime: low-latency (LL) path.
//   - MoEHighThroughputRuntime (exposed as `ExpertParallelRuntime`): the
//     intranode high-throughput (HT) path. Dynamic recv sizing uses an explicit
//     two-phase API (notify_dispatch -> caller allocates -> dispatch); the caller
//     passes device pointers (`tensor.data_ptr()`)
//     and sizes, exactly like the LL runtime.

#include <nanobind/nanobind.h>

#include <cstdint>

#include "api.cuh"
#include "config.hpp"
#include "ht/config.hpp"
#include "ht_runtime.hpp"
#include "moe_runtime.hpp"

namespace nb = nanobind;

namespace {

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  nb::module_::import_("mscclpp._mscclpp");

  nb::enum_<mscclpp::ep::MoEMode>(m, "MoEMode")
      .value("LOW_LATENCY", mscclpp::ep::MoEMode::LOW_LATENCY)
      .value("HIGH_THROUGHPUT", mscclpp::ep::MoEMode::HIGH_THROUGHPUT);

  nb::enum_<mscclpp::ep::DispatchLayout>(m, "DispatchLayout")
      .value("EXPERT_MAJOR", mscclpp::ep::DispatchLayout::EXPERT_MAJOR)
      .value("FLAT", mscclpp::ep::DispatchLayout::FLAT)
      .value("TOKEN_MAJOR", mscclpp::ep::DispatchLayout::TOKEN_MAJOR);

  nb::enum_<mscclpp::ep::low_latency::CombineMode>(m, "CombineMode")
      .value("RANK_LOCAL_REDUCE", mscclpp::ep::low_latency::CombineMode::RANK_LOCAL_REDUCE)
      .value("DIRECT_SEND", mscclpp::ep::low_latency::CombineMode::DIRECT_SEND);
  nb::enum_<mscclpp::ep::low_latency::DispatchDataType>(m, "DispatchDataType")
      .value("BF16", mscclpp::ep::low_latency::DispatchDataType::BF16)
      .value("FP8_E4M3", mscclpp::ep::low_latency::DispatchDataType::FP8_E4M3)
      .value("MXFP8_E4M3", mscclpp::ep::low_latency::DispatchDataType::MXFP8_E4M3);

  nb::class_<mscclpp::ep::MoERuntime>(m, "MoERuntime")
      .def(nb::init<mscclpp::Communicator&, int, int, int, int>(), nb::arg("comm"), nb::arg("max_tokens_per_rank"),
           nb::arg("hidden"), nb::arg("num_experts"), nb::arg("num_topk"))
      .def("is_available", &mscclpp::ep::MoERuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoERuntime::isInternodeAvailable)
      .def(
          "dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t outputPtr, uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputTopkIdxPtr,
             uintptr_t outputTopkWeightsPtr, uintptr_t outputLayoutRangePtr, uintptr_t outputCountPtr, int numTokens,
             int hidden, int numTopk, int maxTokensPerRank, int numExperts, mscclpp::ep::DispatchLayout dispatchLayout,
             mscclpp::ep::low_latency::DispatchDataType dispatchDataType, int numBlocks, uintptr_t streamPtr) {
            self.dispatch(ptr(outputPtr), reinterpret_cast<float*>(ptr(outputScalesPtr)),
                          reinterpret_cast<int*>(ptr(outputSrcInfoPtr)), reinterpret_cast<int*>(ptr(outputTopkIdxPtr)),
                          reinterpret_cast<float*>(ptr(outputTopkWeightsPtr)),
                          reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)),
                          reinterpret_cast<int*>(ptr(outputCountPtr)), ptr(inputPtr),
                          reinterpret_cast<int64_t*>(ptr(topkIdxPtr)), reinterpret_cast<float*>(ptr(topkWeightsPtr)),
                          numTokens, hidden, numTopk, maxTokensPerRank, numExperts, dispatchLayout, dispatchDataType,
                          numBlocks, stream(streamPtr));
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("output_ptr"),
          nb::arg("output_scales_ptr"), nb::arg("output_src_info_ptr"), nb::arg("output_topk_idx_ptr"),
          nb::arg("output_topk_weights_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("max_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("dispatch_layout"), nb::arg("dispatch_data_type"), nb::arg("num_blocks"),
          nb::arg("stream_ptr"))
      .def(
          "combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t srcInfoPtr, uintptr_t layoutRangePtr, uintptr_t outputPtr, int numTokens, int hidden,
             int numTopk, int maxTokensPerRank, int numExperts, mscclpp::ep::DispatchLayout dispatchLayout,
             mscclpp::ep::low_latency::DispatchDataType dispatchDataType, mscclpp::ep::low_latency::CombineMode mode,
             int numBlocks, uintptr_t streamPtr) {
            self.combine(ptr(outputPtr), ptr(expertOutputPtr), reinterpret_cast<int64_t*>(ptr(topkIdxPtr)),
                         reinterpret_cast<float*>(ptr(topkWeightsPtr)), reinterpret_cast<int*>(ptr(srcInfoPtr)),
                         reinterpret_cast<int64_t*>(ptr(layoutRangePtr)), numTokens, hidden, numTopk, maxTokensPerRank,
                         numExperts, dispatchLayout, dispatchDataType, mode, numBlocks, stream(streamPtr));
          },
          nb::arg("expert_output_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("src_info_ptr"),
          nb::arg("layout_range_ptr"), nb::arg("output_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("num_topk"), nb::arg("max_tokens_per_rank"), nb::arg("num_experts"), nb::arg("dispatch_layout"),
          nb::arg("dispatch_data_type"), nb::arg("mode"), nb::arg("num_blocks"), nb::arg("stream_ptr"));

  nb::class_<mscclpp::ep::Config>(m, "Config")
      .def(nb::init<int, int, int>(), nb::arg("num_sms") = 20, nb::arg("num_max_nvl_chunked_send_tokens") = 6,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &mscclpp::ep::Config::get_nvl_buffer_size_hint);

  nb::class_<mscclpp::ep::MoEHighThroughputRuntime>(m, "ExpertParallelRuntime")
      .def(nb::init<mscclpp::Communicator&, int64_t, const mscclpp::ep::Config&>(), nb::arg("comm"),
           nb::arg("max_hidden_bytes"), nb::arg("config"))
      .def("is_available", &mscclpp::ep::MoEHighThroughputRuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoEHighThroughputRuntime::isInternodeAvailable)
      .def(
          "layout",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t num_tokens_per_rank_ptr,
             uintptr_t num_tokens_per_expert_ptr, uintptr_t is_token_in_rank_ptr, uintptr_t topk_idx_ptr,
             int num_tokens, int num_topk, int num_experts, uintptr_t stream_ptr) {
            self.layout(reinterpret_cast<int*>(ptr(num_tokens_per_rank_ptr)),
                        reinterpret_cast<int*>(ptr(num_tokens_per_expert_ptr)),
                        reinterpret_cast<bool*>(ptr(is_token_in_rank_ptr)),
                        reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)), num_tokens, num_topk, num_experts,
                        stream(stream_ptr));
          },
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("num_tokens"), nb::arg("num_topk"), nb::arg("num_experts"),
          nb::arg("stream_ptr"))
      .def("get_dispatch_num_channels", [](const mscclpp::ep::MoEHighThroughputRuntime& self,
                                           int x_element_size) { return self.getDispatchNumChannels(x_element_size); })
      .def("resolve_recv_x_buffer",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self, int num_tokens, int num_recv_tokens, int hidden,
              int x_element_size) -> uintptr_t {
             return reinterpret_cast<uintptr_t>(
                 self.resolveRecvXBuffer(num_tokens, num_recv_tokens, hidden, x_element_size));
           })
      .def(
          "notify_dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t rank_prefix_matrix_ptr,
             uintptr_t channel_prefix_matrix_ptr, uintptr_t num_recv_tokens_per_expert_ptr,
             uintptr_t num_tokens_per_rank_ptr, uintptr_t num_tokens_per_expert_ptr, uintptr_t is_token_in_rank_ptr,
             int num_tokens, int num_experts, int x_element_size, int expert_alignment, uintptr_t stream_ptr) {
            return self.notifyDispatch(reinterpret_cast<int*>(ptr(rank_prefix_matrix_ptr)),
                                       reinterpret_cast<int*>(ptr(channel_prefix_matrix_ptr)),
                                       reinterpret_cast<int*>(ptr(num_recv_tokens_per_expert_ptr)),
                                       reinterpret_cast<const int*>(ptr(num_tokens_per_rank_ptr)),
                                       reinterpret_cast<const int*>(ptr(num_tokens_per_expert_ptr)),
                                       reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)), num_tokens,
                                       num_experts, x_element_size, expert_alignment, stream(stream_ptr));
          },
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"),
          nb::arg("num_recv_tokens_per_expert_ptr"), nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("x_element_size"), nb::arg("expert_alignment"), nb::arg("stream_ptr"))
      .def(
          "dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t recv_x_ptr, uintptr_t recv_x_scales_ptr,
             uintptr_t recv_topk_idx_ptr, uintptr_t recv_topk_weights_ptr, uintptr_t recv_src_idx_ptr,
             uintptr_t send_head_ptr, uintptr_t recv_channel_prefix_matrix_ptr, uintptr_t x_ptr, uintptr_t x_scales_ptr,
             uintptr_t topk_idx_ptr, uintptr_t topk_weights_ptr, uintptr_t is_token_in_rank_ptr,
             uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr, int num_tokens, int hidden,
             int num_topk, int num_scales, int num_experts, int x_element_size, int num_recv_tokens, bool cached_mode,
             uintptr_t stream_ptr) {
            self.dispatch(ptr(recv_x_ptr), reinterpret_cast<float*>(ptr(recv_x_scales_ptr)),
                          reinterpret_cast<int64_t*>(ptr(recv_topk_idx_ptr)),
                          reinterpret_cast<float*>(ptr(recv_topk_weights_ptr)),
                          reinterpret_cast<int*>(ptr(recv_src_idx_ptr)), reinterpret_cast<int*>(ptr(send_head_ptr)),
                          reinterpret_cast<int*>(ptr(recv_channel_prefix_matrix_ptr)), ptr(x_ptr),
                          reinterpret_cast<const float*>(ptr(x_scales_ptr)),
                          reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)),
                          reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                          reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)),
                          reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                          reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)), num_tokens, hidden, num_topk,
                          num_scales, num_experts, x_element_size, num_recv_tokens, cached_mode, stream(stream_ptr));
          },
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"), nb::arg("recv_src_idx_ptr"), nb::arg("send_head_ptr"),
          nb::arg("recv_channel_prefix_matrix_ptr"), nb::arg("x_ptr"), nb::arg("x_scales_ptr"), nb::arg("topk_idx_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"),
          nb::arg("num_scales"), nb::arg("num_experts"), nb::arg("x_element_size"), nb::arg("num_recv_tokens"),
          nb::arg("cached_mode"), nb::arg("stream_ptr"))
      .def(
          "combine",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t combined_x_ptr, uintptr_t combined_topk_weights_ptr,
             uintptr_t x_ptr, uintptr_t topk_weights_ptr, uintptr_t src_idx_ptr, uintptr_t rank_prefix_matrix_ptr,
             uintptr_t channel_prefix_matrix_ptr, uintptr_t send_head_ptr, int num_tokens, int num_recv_tokens,
             int hidden, int num_topk, int x_element_size, int ring_num_channels, uintptr_t stream_ptr) {
            self.combine(ptr(combined_x_ptr), reinterpret_cast<float*>(ptr(combined_topk_weights_ptr)), ptr(x_ptr),
                         reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                         reinterpret_cast<const int*>(ptr(src_idx_ptr)),
                         reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                         reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)),
                         reinterpret_cast<const int*>(ptr(send_head_ptr)), num_tokens, num_recv_tokens, hidden,
                         num_topk, x_element_size, ring_num_channels, stream(stream_ptr));
          },
          nb::arg("combined_x_ptr"), nb::arg("combined_topk_weights_ptr"), nb::arg("x_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("src_idx_ptr"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("send_head_ptr"), nb::arg("num_tokens"),
          nb::arg("num_recv_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("x_element_size"),
          nb::arg("ring_num_channels"), nb::arg("stream_ptr"));
}
