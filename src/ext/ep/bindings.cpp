// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// nanobind module definition for the MSCCL++ EP extension.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "kernels/api.cuh"
#include "moe_runtime.hpp"

namespace nb = nanobind;

namespace {

nb::bytes stringToBytes(const std::string& data) { return nb::bytes(data.data(), data.size()); }

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  m.def("get_low_latency_rdma_size_hint", &mscclpp::ep::getLowLatencyRdmaSizeHint);

  nb::module_::import_("mscclpp._mscclpp");

  nb::enum_<mscclpp::ep::MoEMode>(m, "MoEMode")
      .value("LOW_LATENCY", mscclpp::ep::MoEMode::LOW_LATENCY)
      .value("HIGH_THROUGHPUT", mscclpp::ep::MoEMode::HIGH_THROUGHPUT);

  nb::enum_<mscclpp::ep::DispatchLayout>(m, "DispatchLayout")
      .value("EXPERT_MAJOR", mscclpp::ep::DispatchLayout::EXPERT_MAJOR)
      .value("FLAT", mscclpp::ep::DispatchLayout::FLAT);

  nb::class_<mscclpp::ep::MoERuntime>(m, "MoERuntime")
      .def(nb::init<mscclpp::Communicator&, int64_t, int64_t, mscclpp::ep::MoEMode>(), nb::arg("comm"),
           nb::arg("num_nvl_bytes"), nb::arg("num_rdma_bytes"), nb::arg("mode"))
      .def("is_available", &mscclpp::ep::MoERuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoERuntime::isInternodeAvailable)
      .def("get_num_rdma_ranks", &mscclpp::ep::MoERuntime::getNumRdmaRanks)
      .def("get_rdma_rank", &mscclpp::ep::MoERuntime::getRdmaRank)
      .def("get_root_rdma_rank", &mscclpp::ep::MoERuntime::getRootRdmaRank)
      .def("get_local_device_id", &mscclpp::ep::MoERuntime::getLocalDeviceId)
      .def("get_local_ipc_handle",
           [](const mscclpp::ep::MoERuntime& self) { return stringToBytes(self.getLocalIpcHandle()); })
      .def(
          "dispatch",
          [](mscclpp::ep::MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t outputPtr,
             uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputLayoutRangePtr,
             uintptr_t outputCountPtr, int numTokens, int hidden, int numTopk, int numMaxDispatchTokensPerRank,
             int numExperts, bool useFp8, mscclpp::ep::DispatchLayout outputLayout, uintptr_t streamPtr) {
            self.dispatch(ptr(outputPtr), reinterpret_cast<float*>(ptr(outputScalesPtr)),
                          reinterpret_cast<int*>(ptr(outputSrcInfoPtr)),
                          reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)),
                          reinterpret_cast<int*>(ptr(outputCountPtr)), ptr(inputPtr),
                          reinterpret_cast<int64_t*>(ptr(topkIdxPtr)), numTokens, hidden, numTopk,
                          numMaxDispatchTokensPerRank, numExperts, useFp8, outputLayout, stream(streamPtr));
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("output_ptr"), nb::arg("output_scales_ptr"),
          nb::arg("output_src_info_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_max_dispatch_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("use_fp8"), nb::arg("output_layout"), nb::arg("stream_ptr"))
      .def(
          "combine",
          [](mscclpp::ep::MoERuntime& self, uintptr_t expertOutputPtr, uintptr_t expertScalesPtr, uintptr_t topkIdxPtr,
             uintptr_t topkWeightsPtr, uintptr_t srcInfoPtr, uintptr_t layoutRangePtr, uintptr_t outputPtr,
             int numTokens, int hidden, int numTopk, int numMaxDispatchTokensPerRank, int numExperts,
             bool requiresDequantization, uintptr_t streamPtr) {
            self.combine(ptr(outputPtr), ptr(expertOutputPtr), reinterpret_cast<float*>(ptr(expertScalesPtr)),
                         reinterpret_cast<int64_t*>(ptr(topkIdxPtr)), reinterpret_cast<float*>(ptr(topkWeightsPtr)),
                         reinterpret_cast<int*>(ptr(srcInfoPtr)), reinterpret_cast<int64_t*>(ptr(layoutRangePtr)),
                         numTokens, hidden, numTopk, numMaxDispatchTokensPerRank, numExperts, requiresDequantization,
                         stream(streamPtr));
          },
          nb::arg("expert_output_ptr"), nb::arg("expert_scales_ptr"), nb::arg("topk_idx_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("src_info_ptr"), nb::arg("layout_range_ptr"), nb::arg("output_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_max_dispatch_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("requires_dequantization"), nb::arg("stream_ptr"));
}
