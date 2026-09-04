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
//   - MoEMode.LATENCY selects latency-oriented dispatch/combine.
//   - MoEMode.THROUGHPUT selects bounded-resource receive-pool dispatch/combine.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstdint>
#include <mscclpp/ext/ep/moe_runtime.hpp>

namespace nb = nanobind;

namespace {

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  nb::module_::import_("mscclpp._mscclpp");

  nb::enum_<mscclpp::ep::MoEMode>(m, "MoEMode")
      .value("LATENCY", mscclpp::ep::MoEMode::LATENCY)
      .value("THROUGHPUT", mscclpp::ep::MoEMode::THROUGHPUT);

  nb::enum_<mscclpp::ep::DispatchLayout>(m, "DispatchLayout")
      .value("EXPERT_MAJOR", mscclpp::ep::DispatchLayout::EXPERT_MAJOR)
      .value("KI_RAGGED", mscclpp::ep::DispatchLayout::KI_RAGGED)
      .value("TOKEN_MAJOR", mscclpp::ep::DispatchLayout::TOKEN_MAJOR)
      .value("RANK_MAJOR", mscclpp::ep::DispatchLayout::RANK_MAJOR);

  nb::enum_<mscclpp::ep::CombineMode>(m, "CombineMode")
      .value("RANK_LOCAL_REDUCE", mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE)
      .value("DIRECT_SEND", mscclpp::ep::CombineMode::DIRECT_SEND);
  nb::enum_<mscclpp::ep::DispatchDataType>(m, "DispatchDataType")
      .value("BF16", mscclpp::ep::DispatchDataType::BF16)
      .value("FP16", mscclpp::ep::DispatchDataType::FP16)
      .value("FP8_E4M3", mscclpp::ep::DispatchDataType::FP8_E4M3);

  m.def("create_moe_runtime", &mscclpp::ep::createMoERuntime, nb::arg("comm"), nb::arg("mode"),
        nb::arg("max_tokens_per_rank") = 0, nb::arg("hidden") = 0, nb::arg("num_experts") = 0, nb::arg("num_topk") = 0,
        nb::arg("max_hidden_bytes") = 0, nb::arg("output_layout") = mscclpp::ep::DispatchLayout::EXPERT_MAJOR,
        nb::arg("combine_mode") = mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE,
        "Create the MoE backend selected by mode; returns a shared MoERuntime handle.");

  nb::class_<mscclpp::ep::MoERuntime>(m, "MoERuntime")
      .def_prop_ro("mode", &mscclpp::ep::MoERuntime::mode)
      .def("is_available", &mscclpp::ep::MoERuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoERuntime::isInternodeAvailable)
      .def("initialize", &mscclpp::ep::MoERuntime::initialize)
      .def("output_topk_ids_buffer_ptr",
           [](const mscclpp::ep::MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.outputTopkIdsBuffer()); })
      .def("output_topk_weights_buffer_ptr",
           [](const mscclpp::ep::MoERuntime& self) {
             return reinterpret_cast<uintptr_t>(self.outputTopkWeightsBuffer());
           })
      .def("dispatch_output_buffer_ptr",
           [](const mscclpp::ep::MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.dispatchOutputBuffer()); })
      .def("combine_input_buffer_ptr",
           [](const mscclpp::ep::MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.combineInputBuffer()); })
      .def(
          "dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t outputPtr, uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputTopkIdxPtr,
             uintptr_t outputTopkWeightsPtr, uintptr_t outputLayoutRangePtr, uintptr_t outputCountPtr, int numTokens,
             int hidden, int numTopk, int maxTokensPerRank, int numExperts, int invalidTokenExpertId,
             mscclpp::ep::DispatchLayout dispatchLayout, mscclpp::ep::DispatchDataType dispatchDataType, int numBlocks,
             uintptr_t streamPtr) {
            self.dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{
                .output = ptr(outputPtr),
                .outputScales = ptr(outputScalesPtr),
                .outputSrcInfo = reinterpret_cast<int*>(ptr(outputSrcInfoPtr)),
                .outputTopkIdx = reinterpret_cast<int*>(ptr(outputTopkIdxPtr)),
                .outputTopkWeights = reinterpret_cast<float*>(ptr(outputTopkWeightsPtr)),
                .outputLayoutRange = reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)),
                .outputCount = reinterpret_cast<int*>(ptr(outputCountPtr)),
                .input = ptr(inputPtr),
                .topkIdx = reinterpret_cast<const int64_t*>(ptr(topkIdxPtr)),
                .topkWeights = reinterpret_cast<const float*>(ptr(topkWeightsPtr)),
                .numTokens = numTokens,
                .hidden = hidden,
                .numTopk = numTopk,
                .maxTokensPerRank = maxTokensPerRank,
                .numExperts = numExperts,
                .invalidTokenExpertId = invalidTokenExpertId,
                .dispatchLayout = dispatchLayout,
                .dispatchDataType = dispatchDataType,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
            }});
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("output_ptr"),
          nb::arg("output_scales_ptr"), nb::arg("output_src_info_ptr"), nb::arg("output_topk_idx_ptr"),
          nb::arg("output_topk_weights_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("max_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("invalid_token_expert_id"), nb::arg("dispatch_layout"),
          nb::arg("dispatch_data_type"), nb::arg("num_blocks"), nb::arg("stream_ptr"))
      .def(
          "combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t srcInfoPtr, uintptr_t layoutRangePtr, uintptr_t outputPtr, int numTokens, int hidden,
             int numTopk, int maxTokensPerRank, int numExperts, mscclpp::ep::DispatchLayout dispatchLayout,
             mscclpp::ep::DispatchDataType dispatchDataType, mscclpp::ep::CombineMode mode, int numBlocks,
             uintptr_t streamPtr) {
            self.combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{
                .output = ptr(outputPtr),
                .input = ptr(expertOutputPtr),
                .topkIdx = reinterpret_cast<const int64_t*>(ptr(topkIdxPtr)),
                .topkWeights = reinterpret_cast<const float*>(ptr(topkWeightsPtr)),
                .srcInfo = reinterpret_cast<const int*>(ptr(srcInfoPtr)),
                .layoutRange = reinterpret_cast<const int64_t*>(ptr(layoutRangePtr)),
                .numTokens = numTokens,
                .hidden = hidden,
                .numTopk = numTopk,
                .maxTokensPerRank = maxTokensPerRank,
                .numExperts = numExperts,
                .dispatchLayout = dispatchLayout,
                .dispatchDataType = dispatchDataType,
                .combineMode = mode,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
            }});
          },
          nb::arg("expert_output_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("src_info_ptr"),
          nb::arg("layout_range_ptr"), nb::arg("output_ptr"), nb::arg("num_tokens"), nb::arg("hidden"),
          nb::arg("num_topk"), nb::arg("max_tokens_per_rank"), nb::arg("num_experts"), nb::arg("dispatch_layout"),
          nb::arg("dispatch_data_type"), nb::arg("mode"), nb::arg("num_blocks"), nb::arg("stream_ptr"))
      .def(
          "prepare",
          [](mscclpp::ep::MoERuntime& self, uintptr_t num_tokens_per_rank_ptr, uintptr_t num_tokens_per_expert_ptr,
             uintptr_t is_token_in_rank_ptr, uintptr_t topk_idx_ptr, int num_tokens, int num_topk, int num_experts,
             uintptr_t stream_ptr) {
            self.prepare(reinterpret_cast<int*>(ptr(num_tokens_per_rank_ptr)),
                         reinterpret_cast<int*>(ptr(num_tokens_per_expert_ptr)),
                         reinterpret_cast<bool*>(ptr(is_token_in_rank_ptr)),
                         reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)), num_tokens, num_topk, num_experts,
                         stream(stream_ptr));
          },
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("num_tokens"), nb::arg("num_topk"), nb::arg("num_experts"),
          nb::arg("stream_ptr"))
      .def(
          "notify",
          [](mscclpp::ep::MoERuntime& self, uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr,
             uintptr_t num_recv_tokens_per_expert_ptr, uintptr_t num_tokens_per_rank_ptr,
             uintptr_t num_tokens_per_expert_ptr, uintptr_t is_token_in_rank_ptr, int num_tokens, int num_experts,
             int expert_alignment, int num_blocks, uintptr_t stream_ptr) {
            return self.notify(reinterpret_cast<int*>(ptr(rank_prefix_matrix_ptr)),
                               reinterpret_cast<int*>(ptr(channel_prefix_matrix_ptr)),
                               reinterpret_cast<int*>(ptr(num_recv_tokens_per_expert_ptr)),
                               reinterpret_cast<const int*>(ptr(num_tokens_per_rank_ptr)),
                               reinterpret_cast<const int*>(ptr(num_tokens_per_expert_ptr)),
                               reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)), num_tokens, num_experts,
                               expert_alignment, num_blocks, stream(stream_ptr));
          },
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"),
          nb::arg("num_recv_tokens_per_expert_ptr"), nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("expert_alignment"), nb::arg("num_blocks"), nb::arg("stream_ptr"))
      .def(
          "dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t recv_x_ptr, uintptr_t recv_x_scales_ptr,
             uintptr_t recv_topk_idx_ptr, uintptr_t recv_topk_weights_ptr, uintptr_t send_head_ptr, uintptr_t x_ptr,
             uintptr_t x_scales_ptr, uintptr_t topk_idx_ptr, uintptr_t topk_weights_ptr, uintptr_t is_token_in_rank_ptr,
             uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr, int num_tokens, int hidden,
             int num_topk, int num_scales, int num_experts, int x_element_size, int num_recv_tokens, bool cached_mode,
             int num_blocks, uintptr_t stream_ptr) {
            self.dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::ThroughputDispatchRequest{
                .recvX = ptr(recv_x_ptr),
                .recvXScales = reinterpret_cast<float*>(ptr(recv_x_scales_ptr)),
                .recvTopkIdx = reinterpret_cast<int64_t*>(ptr(recv_topk_idx_ptr)),
                .recvTopkWeights = reinterpret_cast<float*>(ptr(recv_topk_weights_ptr)),
                .sendHead = reinterpret_cast<int*>(ptr(send_head_ptr)),
                .input = ptr(x_ptr),
                .inputScales = reinterpret_cast<const float*>(ptr(x_scales_ptr)),
                .topkIdx = reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)),
                .topkWeights = reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                .isTokenInRank = reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)),
                .rankPrefixMatrix = reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                .channelPrefixMatrix = reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)),
                .numTokens = num_tokens,
                .hidden = hidden,
                .numTopk = num_topk,
                .numScales = num_scales,
                .numExperts = num_experts,
                .inputElementSize = x_element_size,
                .numRecvTokens = num_recv_tokens,
                .cachedMode = cached_mode,
                .numBlocks = num_blocks,
                .stream = stream(stream_ptr),
            }});
          },
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"), nb::arg("send_head_ptr"), nb::arg("x_ptr"), nb::arg("x_scales_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"), nb::arg("num_tokens"),
          nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_scales"), nb::arg("num_experts"),
          nb::arg("x_element_size"), nb::arg("num_recv_tokens"), nb::arg("cached_mode"), nb::arg("num_blocks"),
          nb::arg("stream_ptr"))
      .def(
          "combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t combined_x_ptr, uintptr_t combined_topk_weights_ptr,
             uintptr_t x_ptr, uintptr_t topk_weights_ptr, uintptr_t send_head_ptr, int num_input_tokens,
             int num_output_tokens, int hidden, int num_topk, int x_element_size, int num_blocks,
             uintptr_t stream_ptr) {
            self.combine(mscclpp::ep::CombineRequest{mscclpp::ep::ThroughputCombineRequest{
                .output = ptr(combined_x_ptr),
                .outputTopkWeights = reinterpret_cast<float*>(ptr(combined_topk_weights_ptr)),
                .input = ptr(x_ptr),
                .topkWeights = reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                .sendHead = reinterpret_cast<const int*>(ptr(send_head_ptr)),
                .numInputTokens = num_input_tokens,
                .numOutputTokens = num_output_tokens,
                .hidden = hidden,
                .numTopk = num_topk,
                .inputElementSize = x_element_size,
                .numBlocks = num_blocks,
                .stream = stream(stream_ptr),
            }});
          },
          nb::arg("combined_x_ptr"), nb::arg("combined_topk_weights_ptr"), nb::arg("x_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("send_head_ptr"), nb::arg("num_input_tokens"),
          nb::arg("num_output_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("x_element_size"),
          nb::arg("num_blocks"), nb::arg("stream_ptr"));
}
