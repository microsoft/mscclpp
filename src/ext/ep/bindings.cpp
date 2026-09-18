// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstdint>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <utility>

namespace nb = nanobind;
using namespace mscclpp::ep;

namespace {

template <typename T = void>
T* pointer(uintptr_t address) {
  return reinterpret_cast<T*>(address);
}

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
#ifdef MSCCLPP_DISABLE_NB_LEAK_WARNINGS
  nb::set_leak_warnings(false);
#endif
  m.doc() = "Raw-pointer bindings for the MSCCL++ C++ expert-parallel runtime";
  nb::module_::import_("mscclpp._mscclpp");

  nb::enum_<MoEMode>(m, "MoEMode").value("LATENCY", MoEMode::LATENCY).value("THROUGHPUT", MoEMode::THROUGHPUT);
  nb::enum_<DispatchLayout>(m, "DispatchLayout")
      .value("EXPERT_MAJOR", DispatchLayout::EXPERT_MAJOR)
      .value("TOKEN_MAJOR", DispatchLayout::TOKEN_MAJOR)
      .value("RANK_MAJOR", DispatchLayout::RANK_MAJOR);
  nb::enum_<CombineMode>(m, "CombineMode")
      .value("RANK_LOCAL_REDUCE", CombineMode::RANK_LOCAL_REDUCE)
      .value("DIRECT_SEND", CombineMode::DIRECT_SEND);
  nb::enum_<DispatchDataType>(m, "DispatchDataType")
      .value("BF16", DispatchDataType::BF16)
      .value("FP8_E4M3", DispatchDataType::FP8_E4M3);

  nb::class_<PrepareHandle>(m, "PrepareHandle").def(nb::init<>());
  nb::class_<DispatchHandle>(m, "DispatchHandle").def(nb::init<>());

  m.def("create_moe_runtime", &createMoERuntime, nb::arg("comm"), nb::arg("mode"), nb::arg("max_tokens_per_rank"),
        nb::arg("hidden"), nb::arg("num_experts"), nb::arg("num_topk"),
        nb::arg("output_layout") = DispatchLayout::EXPERT_MAJOR,
        nb::arg("combine_mode") = CombineMode::RANK_LOCAL_REDUCE, nb::keep_alive<0, 1>());

  nb::class_<MoERuntime>(m, "MoERuntime")
      .def_prop_ro("mode", &MoERuntime::mode)
      .def_prop_ro("rank", &MoERuntime::rank)
      .def_prop_ro("num_ranks", &MoERuntime::numRanks)
      .def_prop_ro("num_ranks_per_ipc_domain", &MoERuntime::numRanksPerIpcDomain)
      .def("is_available", &MoERuntime::isAvailable)
      .def("initialize", &MoERuntime::initialize, nb::call_guard<nb::gil_scoped_release>())
      .def("dispatch_output_buffer_ptr",
           [](const MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.dispatchOutputBuffer()); })
      .def("combine_input_buffer_ptr",
           [](const MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.combineInputBuffer()); })
      .def("output_topk_ids_buffer_ptr",
           [](const MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.outputTopkIdsBuffer()); })
      .def("output_topk_weights_buffer_ptr",
           [](const MoERuntime& self) { return reinterpret_cast<uintptr_t>(self.outputTopkWeightsBuffer()); })
      .def(
          "prepare",
          [](MoERuntime& self, uintptr_t topkIdxPtr, int numTokens, int maxTokensPerRank, int numBlocks,
             uintptr_t streamPtr) {
            return self.prepare(
                {pointer<const int64_t>(topkIdxPtr), numTokens, maxTokensPerRank, numBlocks, stream(streamPtr)});
          },
          nb::arg("topk_idx_ptr"), nb::arg("num_tokens"), nb::arg("max_tokens_per_rank"), nb::arg("num_blocks"),
          nb::arg("stream_ptr"), nb::keep_alive<0, 1>())
      .def(
          "dispatch_latency",
          [](MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr, uintptr_t outputPtr,
             uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputTopkIdxPtr,
             uintptr_t outputTopkWeightsPtr, uintptr_t outputLayoutRangePtr, uintptr_t outputCountPtr, int numTokens,
             int maxTokensPerRank, int invalidTokenExpertId, DispatchDataType dataType, int numBlocks,
             uintptr_t streamPtr) {
            return self.dispatch(DispatchRequest{LatencyDispatchRequest{
                .output = pointer(outputPtr),
                .outputScales = pointer(outputScalesPtr),
                .outputSrcInfo = pointer<int>(outputSrcInfoPtr),
                .outputTopkIdx = pointer<int>(outputTopkIdxPtr),
                .outputTopkWeights = pointer<float>(outputTopkWeightsPtr),
                .outputLayoutRange = pointer<int64_t>(outputLayoutRangePtr),
                .outputCount = pointer<int>(outputCountPtr),
                .input = pointer<const void>(inputPtr),
                .topkIdx = pointer<const int64_t>(topkIdxPtr),
                .topkWeights = pointer<const float>(topkWeightsPtr),
                .numTokens = numTokens,
                .maxTokensPerRank = maxTokensPerRank,
                .invalidTokenExpertId = invalidTokenExpertId,
                .dispatchDataType = dataType,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
            }});
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("output_ptr"),
          nb::arg("output_scales_ptr"), nb::arg("output_src_info_ptr"), nb::arg("output_topk_idx_ptr"),
          nb::arg("output_topk_weights_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("max_tokens_per_rank"), nb::arg("invalid_token_expert_id"),
          nb::arg("dispatch_data_type"), nb::arg("num_blocks"), nb::arg("stream_ptr"), nb::keep_alive<0, 1>())
      .def(
          "dispatch_throughput",
          [](MoERuntime& self, uintptr_t inputPtr, uintptr_t inputScalesPtr, uintptr_t topkIdxPtr,
             uintptr_t topkWeightsPtr, uintptr_t outputPtr, uintptr_t outputScalesPtr, uintptr_t outputTopkIdxPtr,
             uintptr_t outputTopkWeightsPtr, uintptr_t outputCountPtr, int numTokens, int maxTokensPerRank,
             DispatchDataType dataType, int numBlocks, uintptr_t streamPtr, PrepareHandle preparation) {
            return self.dispatch(DispatchRequest{ThroughputDispatchRequest{
                .output = pointer(outputPtr),
                .outputScales = pointer(outputScalesPtr),
                .outputTopkIdx = pointer<int>(outputTopkIdxPtr),
                .outputTopkWeights = pointer<float>(outputTopkWeightsPtr),
                .outputCount = pointer<int>(outputCountPtr),
                .input = pointer<const void>(inputPtr),
                .inputScales = pointer<const float>(inputScalesPtr),
                .topkIdx = pointer<const int64_t>(topkIdxPtr),
                .topkWeights = pointer<const float>(topkWeightsPtr),
                .numTokens = numTokens,
                .maxTokensPerRank = maxTokensPerRank,
                .dispatchDataType = dataType,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
                .prepareHandle = std::move(preparation),
            }});
          },
          nb::arg("input_ptr"), nb::arg("input_scales_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"),
          nb::arg("output_ptr"), nb::arg("output_scales_ptr"), nb::arg("output_topk_idx_ptr"),
          nb::arg("output_topk_weights_ptr"), nb::arg("output_count_ptr"), nb::arg("num_tokens"),
          nb::arg("max_tokens_per_rank"), nb::arg("dispatch_data_type"), nb::arg("num_blocks"), nb::arg("stream_ptr"),
          nb::arg("prepare_handle") = PrepareHandle{}, nb::keep_alive<0, 1>())
      .def(
          "combine_latency",
          [](MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t outputPtr, const DispatchHandle& handle,
             int numBlocks, uintptr_t streamPtr) {
            self.combine(CombineRequest{LatencyCombineRequest{
                .output = pointer(outputPtr),
                .input = pointer<const void>(expertOutputPtr),
                .handle = handle,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
            }});
          },
          nb::arg("expert_output_ptr"), nb::arg("output_ptr"), nb::arg("handle"), nb::arg("num_blocks"),
          nb::arg("stream_ptr"))
      .def(
          "combine_throughput",
          [](MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t outputPtr, uintptr_t outputTopkWeightsPtr,
             const DispatchHandle& handle, int numBlocks, uintptr_t streamPtr) {
            self.combine(CombineRequest{ThroughputCombineRequest{
                .output = pointer(outputPtr),
                .outputTopkWeights = pointer<float>(outputTopkWeightsPtr),
                .input = pointer<const void>(expertOutputPtr),
                .handle = handle,
                .numBlocks = numBlocks,
                .stream = stream(streamPtr),
            }});
          },
          nb::arg("expert_output_ptr"), nb::arg("output_ptr"), nb::arg("output_topk_weights_ptr"), nb::arg("handle"),
          nb::arg("num_blocks"), nb::arg("stream_ptr"));
}
