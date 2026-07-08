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
//     high-throughput (HT) DeepEP-style path. Dynamic recv sizing uses an
//     explicit two-phase API (intranode_notify_dispatch -> caller allocates ->
//     intranode_dispatch); the caller passes device pointers (`tensor.data_ptr()`)
//     and sizes, exactly like the LL runtime.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "ht/config.hpp"
#include "ht_runtime.hpp"
#include "kernels/api.cuh"
#include "moe_runtime.hpp"

namespace nb = nanobind;

namespace {

nb::bytes stringToBytes(const std::string& data) { return nb::bytes(data.data(), data.size()); }

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

// Python bytes / bytearray -> std::string (raw bytes).
std::string bytesToString(nb::handle h) {
  PyObject* o = h.ptr();
  if (PyBytes_Check(o)) {
    char* buf = nullptr;
    Py_ssize_t len = 0;
    PyBytes_AsStringAndSize(o, &buf, &len);
    return std::string(buf, static_cast<size_t>(len));
  }
  if (PyByteArray_Check(o)) {
    return std::string(PyByteArray_AsString(o), static_cast<size_t>(PyByteArray_Size(o)));
  }
  throw std::runtime_error("mscclpp_ep: expected bytes or bytearray");
}

}  // namespace

NB_MODULE(mscclpp_ep_cpp, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  m.def("get_low_latency_rdma_size_hint", &mscclpp::ep::getLowLatencyRdmaSizeHint,
        nb::arg("num_max_dispatch_tokens_per_rank"), nb::arg("hidden"), nb::arg("num_ranks"), nb::arg("num_experts"),
        nb::arg("num_topk"));

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
          [](mscclpp::ep::MoERuntime& self, uintptr_t inputPtr, uintptr_t topkIdxPtr, uintptr_t topkWeightsPtr,
             uintptr_t outputPtr, uintptr_t outputScalesPtr, uintptr_t outputSrcInfoPtr, uintptr_t outputLayoutRangePtr,
             uintptr_t outputCountPtr, int numTokens, int hidden, int numTopk, int numMaxDispatchTokensPerRank,
             int numExperts, bool requiresQuantization, mscclpp::ep::DispatchLayout outputLayout, uintptr_t streamPtr) {
            self.dispatch(
                ptr(outputPtr), reinterpret_cast<float*>(ptr(outputScalesPtr)),
                reinterpret_cast<int*>(ptr(outputSrcInfoPtr)), reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)),
                reinterpret_cast<int*>(ptr(outputCountPtr)), ptr(inputPtr), reinterpret_cast<int64_t*>(ptr(topkIdxPtr)),
                reinterpret_cast<float*>(ptr(topkWeightsPtr)), numTokens, hidden, numTopk, numMaxDispatchTokensPerRank,
                numExperts, requiresQuantization, outputLayout, stream(streamPtr));
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("output_ptr"),
          nb::arg("output_scales_ptr"), nb::arg("output_src_info_ptr"), nb::arg("output_layout_range_ptr"),
          nb::arg("output_count_ptr"), nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"),
          nb::arg("num_max_dispatch_tokens_per_rank"), nb::arg("num_experts"), nb::arg("requires_quantization"),
          nb::arg("output_layout"), nb::arg("stream_ptr"))
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

  // ==========================================================================
  // High-throughput (HT) DeepEP-style backend: Config + MoEHighThroughputRuntime.
  // Torch-free, raw-pointer boundary (uintptr_t == device pointer). Dynamic recv
  // sizing uses the two-phase notify -> (caller allocates) -> dispatch API.
  // ==========================================================================

  nb::class_<mscclpp::ep::Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 20, nb::arg("num_max_nvl_chunked_send_tokens") = 6,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 256, nb::arg("num_max_rdma_chunked_send_tokens") = 6,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &mscclpp::ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint", &mscclpp::ep::Config::get_rdma_buffer_size_hint);

  nb::class_<mscclpp::ep::MoEHighThroughputRuntime>(m, "ExpertParallelRuntime")
      .def(nb::init<int, int, int64_t, int64_t>(), nb::arg("rank"), nb::arg("num_ranks"), nb::arg("num_nvl_bytes"),
           nb::arg("num_rdma_bytes"))
      .def("is_available", &mscclpp::ep::MoEHighThroughputRuntime::isAvailable)
      .def("is_internode_available", &mscclpp::ep::MoEHighThroughputRuntime::isInternodeAvailable)
      .def("get_num_rdma_ranks", &mscclpp::ep::MoEHighThroughputRuntime::getNumRdmaRanks)
      .def("get_rdma_rank", &mscclpp::ep::MoEHighThroughputRuntime::getRdmaRank)
      .def("get_root_rdma_rank", &mscclpp::ep::MoEHighThroughputRuntime::getRootRdmaRank)
      .def("get_local_device_id", &mscclpp::ep::MoEHighThroughputRuntime::getLocalDeviceId)
      .def("get_local_ipc_handle",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self) { return stringToBytes(self.getLocalIpcHandle()); })
      .def("create_unique_id",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self) {
             auto uid = self.createUniqueId();
             return nb::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
           })
      .def("connect",
           [](mscclpp::ep::MoEHighThroughputRuntime& self, nb::object data) {
             std::string s = bytesToString(data);
             mscclpp::UniqueId uid;
             if (s.size() != uid.size()) {
               throw std::runtime_error("mscclpp_ep_cpp.ExpertParallelRuntime.connect: UniqueId size mismatch");
             }
             std::memcpy(uid.data(), s.data(), s.size());
             self.connect(uid);
           })
      .def(
          "sync",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, const std::vector<int>& device_ids,
             nb::sequence all_gathered_handles, nb::object root_unique_id_opt) {
            std::vector<std::optional<std::string>> handles;
            for (nb::handle h : all_gathered_handles) {
              if (h.is_none())
                handles.emplace_back(std::nullopt);
              else
                handles.emplace_back(bytesToString(h));
            }
            std::optional<std::string> root_uid = root_unique_id_opt.is_none()
                                                      ? std::nullopt
                                                      : std::optional<std::string>(bytesToString(root_unique_id_opt));
            self.sync(device_ids, handles, root_uid);
          },
          nb::arg("device_ids"), nb::arg("all_gathered_handles").none(), nb::arg("root_unique_id").none())
      .def(
          "get_dispatch_layout",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t num_tokens_per_rank_ptr,
             uintptr_t num_tokens_per_rdma_rank_ptr, uintptr_t num_tokens_per_expert_ptr,
             uintptr_t is_token_in_rank_ptr, uintptr_t topk_idx_ptr, int num_tokens, int num_topk, int num_experts,
             uintptr_t stream_ptr) {
            self.getDispatchLayout(reinterpret_cast<int*>(ptr(num_tokens_per_rank_ptr)),
                                   reinterpret_cast<int*>(ptr(num_tokens_per_rdma_rank_ptr)),
                                   reinterpret_cast<int*>(ptr(num_tokens_per_expert_ptr)),
                                   reinterpret_cast<bool*>(ptr(is_token_in_rank_ptr)),
                                   reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)), num_tokens, num_topk,
                                   num_experts, stream(stream_ptr));
          },
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("num_tokens_per_rdma_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("topk_idx_ptr"),
          nb::arg("num_tokens"), nb::arg("num_topk"), nb::arg("num_experts"), nb::arg("stream_ptr"))
      .def(
          "get_intranode_dispatch_num_channels",
          [](const mscclpp::ep::MoEHighThroughputRuntime& self, int x_element_size, const mscclpp::ep::Config& config) {
            return self.getIntranodeDispatchNumChannels(x_element_size, config);
          })
      .def("resolve_intranode_recv_x_buffer",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self, int num_recv_tokens, int hidden, int x_element_size,
              const mscclpp::ep::Config& config) -> uintptr_t {
             return reinterpret_cast<uintptr_t>(
                 self.resolveIntranodeRecvXBuffer(num_recv_tokens, hidden, x_element_size, config));
           })
      .def("resolve_internode_recv_x_buffer",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self, int num_recv_tokens, int hidden, int x_element_size,
              const mscclpp::ep::Config& config) -> uintptr_t {
             return reinterpret_cast<uintptr_t>(
                 self.resolveInternodeRecvXBuffer(num_recv_tokens, hidden, x_element_size, config));
           })
      .def("get_internode_dispatch_num_channels",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self, const mscclpp::ep::Config& config) {
             return self.getInternodeDispatchNumChannels(config);
           })
      .def("get_source_meta_bytes",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self) { return self.getSourceMetaBytes(); })
      .def("get_num_max_nvl_peers",
           [](const mscclpp::ep::MoEHighThroughputRuntime& self) { return self.getNumMaxNvlPeers(); })
      .def(
          "intranode_notify_dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t rank_prefix_matrix_ptr,
             uintptr_t channel_prefix_matrix_ptr, uintptr_t num_recv_tokens_per_expert_ptr,
             uintptr_t num_tokens_per_rank_ptr, uintptr_t num_tokens_per_expert_ptr, uintptr_t is_token_in_rank_ptr,
             int num_tokens, int num_experts, int x_element_size, int expert_alignment,
             const mscclpp::ep::Config& config, uintptr_t stream_ptr) {
            return self.intranodeNotifyDispatch(reinterpret_cast<int*>(ptr(rank_prefix_matrix_ptr)),
                                                reinterpret_cast<int*>(ptr(channel_prefix_matrix_ptr)),
                                                reinterpret_cast<int*>(ptr(num_recv_tokens_per_expert_ptr)),
                                                reinterpret_cast<const int*>(ptr(num_tokens_per_rank_ptr)),
                                                reinterpret_cast<const int*>(ptr(num_tokens_per_expert_ptr)),
                                                reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)), num_tokens,
                                                num_experts, x_element_size, expert_alignment, config,
                                                stream(stream_ptr));
          },
          nb::arg("rank_prefix_matrix_ptr"), nb::arg("channel_prefix_matrix_ptr"),
          nb::arg("num_recv_tokens_per_expert_ptr"), nb::arg("num_tokens_per_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("x_element_size"), nb::arg("expert_alignment"), nb::arg("config"),
          nb::arg("stream_ptr"))
      .def(
          "intranode_dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t recv_x_ptr, uintptr_t recv_x_scales_ptr,
             uintptr_t recv_topk_idx_ptr, uintptr_t recv_topk_weights_ptr, uintptr_t recv_src_idx_ptr,
             uintptr_t send_head_ptr, uintptr_t recv_channel_prefix_matrix_ptr, uintptr_t x_ptr, uintptr_t x_scales_ptr,
             uintptr_t topk_idx_ptr, uintptr_t topk_weights_ptr, uintptr_t is_token_in_rank_ptr,
             uintptr_t rank_prefix_matrix_ptr, uintptr_t channel_prefix_matrix_ptr, int num_tokens, int hidden,
             int num_topk, int num_scales, int num_experts, int x_element_size, int num_recv_tokens, bool cached_mode,
             const mscclpp::ep::Config& config, uintptr_t stream_ptr) {
            self.intranodeDispatch(
                ptr(recv_x_ptr), reinterpret_cast<float*>(ptr(recv_x_scales_ptr)),
                reinterpret_cast<int64_t*>(ptr(recv_topk_idx_ptr)),
                reinterpret_cast<float*>(ptr(recv_topk_weights_ptr)), reinterpret_cast<int*>(ptr(recv_src_idx_ptr)),
                reinterpret_cast<int*>(ptr(send_head_ptr)), reinterpret_cast<int*>(ptr(recv_channel_prefix_matrix_ptr)),
                ptr(x_ptr), reinterpret_cast<const float*>(ptr(x_scales_ptr)),
                reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)),
                reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)),
                reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)), num_tokens, hidden, num_topk, num_scales,
                num_experts, x_element_size, num_recv_tokens, cached_mode, config, stream(stream_ptr));
          },
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"), nb::arg("recv_src_idx_ptr"), nb::arg("send_head_ptr"),
          nb::arg("recv_channel_prefix_matrix_ptr"), nb::arg("x_ptr"), nb::arg("x_scales_ptr"), nb::arg("topk_idx_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"),
          nb::arg("num_scales"), nb::arg("num_experts"), nb::arg("x_element_size"), nb::arg("num_recv_tokens"),
          nb::arg("cached_mode"), nb::arg("config"), nb::arg("stream_ptr"))
      .def(
          "intranode_combine",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t combined_x_ptr, uintptr_t combined_topk_weights_ptr,
             uintptr_t x_ptr, uintptr_t topk_weights_ptr, uintptr_t src_idx_ptr, uintptr_t rank_prefix_matrix_ptr,
             uintptr_t channel_prefix_matrix_ptr, uintptr_t send_head_ptr, int num_tokens, int num_recv_tokens,
             int hidden, int num_topk, int x_element_size, int ring_num_channels, const mscclpp::ep::Config& config,
             uintptr_t stream_ptr) {
            self.intranodeCombine(ptr(combined_x_ptr), reinterpret_cast<float*>(ptr(combined_topk_weights_ptr)),
                                  ptr(x_ptr), reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                                  reinterpret_cast<const int*>(ptr(src_idx_ptr)),
                                  reinterpret_cast<const int*>(ptr(rank_prefix_matrix_ptr)),
                                  reinterpret_cast<const int*>(ptr(channel_prefix_matrix_ptr)),
                                  reinterpret_cast<const int*>(ptr(send_head_ptr)), num_tokens, num_recv_tokens, hidden,
                                  num_topk, x_element_size, ring_num_channels, config, stream(stream_ptr));
          },
          nb::arg("combined_x_ptr"), nb::arg("combined_topk_weights_ptr"), nb::arg("x_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("src_idx_ptr"), nb::arg("rank_prefix_matrix_ptr"),
          nb::arg("channel_prefix_matrix_ptr"), nb::arg("send_head_ptr"), nb::arg("num_tokens"),
          nb::arg("num_recv_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("x_element_size"),
          nb::arg("ring_num_channels"), nb::arg("config"), nb::arg("stream_ptr"))
      .def(
          "internode_notify_dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t rdma_channel_prefix_matrix_ptr,
             uintptr_t recv_rdma_rank_prefix_sum_ptr, uintptr_t gbl_channel_prefix_matrix_ptr,
             uintptr_t recv_gbl_rank_prefix_sum_ptr, uintptr_t num_recv_tokens_per_expert_ptr,
             uintptr_t num_rdma_recv_tokens_ptr, uintptr_t num_tokens_per_rank_ptr,
             uintptr_t num_tokens_per_rdma_rank_ptr, uintptr_t num_tokens_per_expert_ptr,
             uintptr_t is_token_in_rank_ptr, int num_tokens, int num_experts, int hidden, int num_scales, int num_topk,
             int x_element_size, int expert_alignment, const mscclpp::ep::Config& config, uintptr_t stream_ptr) {
            return self.internodeNotifyDispatch(reinterpret_cast<int*>(ptr(rdma_channel_prefix_matrix_ptr)),
                                                reinterpret_cast<int*>(ptr(recv_rdma_rank_prefix_sum_ptr)),
                                                reinterpret_cast<int*>(ptr(gbl_channel_prefix_matrix_ptr)),
                                                reinterpret_cast<int*>(ptr(recv_gbl_rank_prefix_sum_ptr)),
                                                reinterpret_cast<int*>(ptr(num_recv_tokens_per_expert_ptr)),
                                                reinterpret_cast<int*>(ptr(num_rdma_recv_tokens_ptr)),
                                                reinterpret_cast<const int*>(ptr(num_tokens_per_rank_ptr)),
                                                reinterpret_cast<const int*>(ptr(num_tokens_per_rdma_rank_ptr)),
                                                reinterpret_cast<const int*>(ptr(num_tokens_per_expert_ptr)),
                                                reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)), num_tokens,
                                                num_experts, hidden, num_scales, num_topk, x_element_size,
                                                expert_alignment, config, stream(stream_ptr));
          },
          nb::arg("rdma_channel_prefix_matrix_ptr"), nb::arg("recv_rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"), nb::arg("recv_gbl_rank_prefix_sum_ptr"),
          nb::arg("num_recv_tokens_per_expert_ptr"), nb::arg("num_rdma_recv_tokens_ptr"),
          nb::arg("num_tokens_per_rank_ptr"), nb::arg("num_tokens_per_rdma_rank_ptr"),
          nb::arg("num_tokens_per_expert_ptr"), nb::arg("is_token_in_rank_ptr"), nb::arg("num_tokens"),
          nb::arg("num_experts"), nb::arg("hidden"), nb::arg("num_scales"), nb::arg("num_topk"),
          nb::arg("x_element_size"), nb::arg("expert_alignment"), nb::arg("config"), nb::arg("stream_ptr"))
      .def(
          "internode_dispatch",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t recv_x_ptr, uintptr_t recv_x_scales_ptr,
             uintptr_t recv_topk_idx_ptr, uintptr_t recv_topk_weights_ptr, uintptr_t recv_src_meta_ptr,
             uintptr_t recv_rdma_channel_prefix_matrix_ptr, uintptr_t recv_gbl_channel_prefix_matrix_ptr,
             uintptr_t send_rdma_head_ptr, uintptr_t send_nvl_head_ptr, uintptr_t x_ptr, uintptr_t x_scales_ptr,
             uintptr_t topk_idx_ptr, uintptr_t topk_weights_ptr, uintptr_t is_token_in_rank_ptr,
             uintptr_t rdma_channel_prefix_matrix_ptr, uintptr_t recv_rdma_rank_prefix_sum_ptr,
             uintptr_t gbl_channel_prefix_matrix_ptr, uintptr_t recv_gbl_rank_prefix_sum_ptr, int num_tokens,
             int hidden, int num_topk, int num_scales, int num_experts, int x_element_size, int num_recv_tokens,
             int num_rdma_recv_tokens, bool cached_mode, const mscclpp::ep::Config& config, uintptr_t stream_ptr) {
            self.internodeDispatch(ptr(recv_x_ptr), reinterpret_cast<float*>(ptr(recv_x_scales_ptr)),
                                   reinterpret_cast<int64_t*>(ptr(recv_topk_idx_ptr)),
                                   reinterpret_cast<float*>(ptr(recv_topk_weights_ptr)), ptr(recv_src_meta_ptr),
                                   reinterpret_cast<int*>(ptr(recv_rdma_channel_prefix_matrix_ptr)),
                                   reinterpret_cast<int*>(ptr(recv_gbl_channel_prefix_matrix_ptr)),
                                   reinterpret_cast<int*>(ptr(send_rdma_head_ptr)),
                                   reinterpret_cast<int*>(ptr(send_nvl_head_ptr)), ptr(x_ptr),
                                   reinterpret_cast<const float*>(ptr(x_scales_ptr)),
                                   reinterpret_cast<const int64_t*>(ptr(topk_idx_ptr)),
                                   reinterpret_cast<const float*>(ptr(topk_weights_ptr)),
                                   reinterpret_cast<const bool*>(ptr(is_token_in_rank_ptr)),
                                   reinterpret_cast<const int*>(ptr(rdma_channel_prefix_matrix_ptr)),
                                   reinterpret_cast<const int*>(ptr(recv_rdma_rank_prefix_sum_ptr)),
                                   reinterpret_cast<const int*>(ptr(gbl_channel_prefix_matrix_ptr)),
                                   reinterpret_cast<const int*>(ptr(recv_gbl_rank_prefix_sum_ptr)), num_tokens, hidden,
                                   num_topk, num_scales, num_experts, x_element_size, num_recv_tokens,
                                   num_rdma_recv_tokens, cached_mode, config, stream(stream_ptr));
          },
          nb::arg("recv_x_ptr"), nb::arg("recv_x_scales_ptr"), nb::arg("recv_topk_idx_ptr"),
          nb::arg("recv_topk_weights_ptr"), nb::arg("recv_src_meta_ptr"),
          nb::arg("recv_rdma_channel_prefix_matrix_ptr"), nb::arg("recv_gbl_channel_prefix_matrix_ptr"),
          nb::arg("send_rdma_head_ptr"), nb::arg("send_nvl_head_ptr"), nb::arg("x_ptr"), nb::arg("x_scales_ptr"),
          nb::arg("topk_idx_ptr"), nb::arg("topk_weights_ptr"), nb::arg("is_token_in_rank_ptr"),
          nb::arg("rdma_channel_prefix_matrix_ptr"), nb::arg("recv_rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"), nb::arg("recv_gbl_rank_prefix_sum_ptr"), nb::arg("num_tokens"),
          nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_scales"), nb::arg("num_experts"),
          nb::arg("x_element_size"), nb::arg("num_recv_tokens"), nb::arg("num_rdma_recv_tokens"),
          nb::arg("cached_mode"), nb::arg("config"), nb::arg("stream_ptr"))
      .def(
          "internode_combine",
          [](mscclpp::ep::MoEHighThroughputRuntime& self, uintptr_t combined_x_ptr, uintptr_t combined_topk_weights_ptr,
             uintptr_t x_ptr, uintptr_t topk_weights_ptr, uintptr_t src_meta_ptr,
             uintptr_t is_combined_token_in_rank_ptr, uintptr_t rdma_channel_prefix_matrix_ptr,
             uintptr_t rdma_rank_prefix_sum_ptr, uintptr_t gbl_channel_prefix_matrix_ptr,
             uintptr_t combined_rdma_head_ptr, uintptr_t combined_nvl_head_ptr, int num_tokens, int num_combined_tokens,
             int hidden, int num_topk, int x_element_size, const mscclpp::ep::Config& config, uintptr_t stream_ptr) {
            self.internodeCombine(ptr(combined_x_ptr), reinterpret_cast<float*>(ptr(combined_topk_weights_ptr)),
                                  ptr(x_ptr), reinterpret_cast<const float*>(ptr(topk_weights_ptr)), ptr(src_meta_ptr),
                                  reinterpret_cast<const bool*>(ptr(is_combined_token_in_rank_ptr)),
                                  reinterpret_cast<const int*>(ptr(rdma_channel_prefix_matrix_ptr)),
                                  reinterpret_cast<const int*>(ptr(rdma_rank_prefix_sum_ptr)),
                                  reinterpret_cast<const int*>(ptr(gbl_channel_prefix_matrix_ptr)),
                                  reinterpret_cast<const int*>(ptr(combined_rdma_head_ptr)),
                                  reinterpret_cast<const int*>(ptr(combined_nvl_head_ptr)), num_tokens,
                                  num_combined_tokens, hidden, num_topk, x_element_size, config, stream(stream_ptr));
          },
          nb::arg("combined_x_ptr"), nb::arg("combined_topk_weights_ptr"), nb::arg("x_ptr"),
          nb::arg("topk_weights_ptr"), nb::arg("src_meta_ptr"), nb::arg("is_combined_token_in_rank_ptr"),
          nb::arg("rdma_channel_prefix_matrix_ptr"), nb::arg("rdma_rank_prefix_sum_ptr"),
          nb::arg("gbl_channel_prefix_matrix_ptr"), nb::arg("combined_rdma_head_ptr"), nb::arg("combined_nvl_head_ptr"),
          nb::arg("num_tokens"), nb::arg("num_combined_tokens"), nb::arg("hidden"), nb::arg("num_topk"),
          nb::arg("x_element_size"), nb::arg("config"), nb::arg("stream_ptr"));
}
