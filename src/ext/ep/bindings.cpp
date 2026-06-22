// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// nanobind module definition for the MSCCL++ EP extension.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "config.hpp"

namespace nb = nanobind;

namespace nanobind::detail {

template <>
struct type_caster<torch::Tensor> {
  NB_TYPE_CASTER(torch::Tensor, const_name("torch.Tensor"));

  bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
    bool isTensor = false;
    try {
      isTensor = THPVariable_Check(src.ptr());
    } catch (...) {
      return false;
    }
    if (!isTensor) return false;
    value = THPVariable_Unpack(src.ptr());
    return true;
  }

  static handle from_cpp(const torch::Tensor& tensor, rv_policy, cleanup_list*) noexcept {
    return handle(THPVariable_Wrap(tensor));
  }
};

}  // namespace nanobind::detail

namespace {

std::string bytesToString(nb::handle data) {
  Py_buffer view;
  if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0) throw nb::python_error();
  std::string result(static_cast<const char*>(view.buf), static_cast<size_t>(view.len));
  PyBuffer_Release(&view);
  return result;
}

nb::bytes stringToBytes(const std::string& data) { return nb::bytes(data.data(), data.size()); }

torch::ScalarType dtypeFromPython(nb::handle dtype) {
  if (!THPDtype_Check(dtype.ptr())) throw nb::type_error("expected torch.dtype");
  return reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type;
}

std::optional<torch::Tensor> optionalTensor(nb::handle obj) {
  if (obj.is_none()) return std::nullopt;
  return nb::cast<torch::Tensor>(obj);
}

std::vector<std::optional<std::string>> optionalBytesList(nb::sequence values) {
  std::vector<std::optional<std::string>> result;
  result.reserve(static_cast<size_t>(nb::len(values)));
  for (size_t i = 0; i < static_cast<size_t>(nb::len(values)); ++i) {
    nb::object item = values[i];
    if (item.is_none()) {
      result.push_back(std::nullopt);
    } else {
      result.push_back(bytesToString(item));
    }
  }
  return result;
}

std::optional<std::string> optionalBytes(nb::handle value) {
  if (value.is_none()) return std::nullopt;
  return bytesToString(value);
}

}  // namespace

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  nb::class_<mscclpp::ep::Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 20, nb::arg("num_max_nvl_chunked_send_tokens") = 6,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 256, nb::arg("num_max_rdma_chunked_send_tokens") = 6,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &mscclpp::ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint", &mscclpp::ep::Config::get_rdma_buffer_size_hint);

  m.def("get_low_latency_rdma_size_hint", &mscclpp::ep::get_low_latency_rdma_size_hint);

  nb::class_<mscclpp::ep::EventHandle>(m, "EventHandle")
      .def(nb::init<>())
      .def("current_stream_wait", &mscclpp::ep::EventHandle::current_stream_wait);

  nb::class_<mscclpp::ep::Buffer>(m, "ExpertParallelRuntime")
      .def(nb::init<int, int, int64_t, int64_t, bool>(), nb::arg("rank"), nb::arg("num_ranks"),
           nb::arg("num_nvl_bytes"), nb::arg("num_rdma_bytes"), nb::arg("low_latency_mode"))
      .def("is_available", &mscclpp::ep::Buffer::is_available)
      .def("is_internode_available", &mscclpp::ep::Buffer::is_internode_available)
      .def("get_num_rdma_ranks", &mscclpp::ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &mscclpp::ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &mscclpp::ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &mscclpp::ep::Buffer::get_local_device_id)
      .def("get_local_ipc_handle",
           [](const mscclpp::ep::Buffer& self) { return stringToBytes(self.get_local_ipc_handle()); })
      .def("get_local_nvshmem_unique_id",
           [](const mscclpp::ep::Buffer& self) { return stringToBytes(self.get_local_nvshmem_unique_id()); })
      .def("get_local_buffer_tensor",
           [](const mscclpp::ep::Buffer& self, nb::handle dtype, int64_t offset, bool useRdmaBuffer) {
             return self.get_local_buffer_tensor(dtypeFromPython(dtype), offset, useRdmaBuffer);
           })
      .def("create_unique_id",
           [](const mscclpp::ep::Buffer& self) {
             auto uid = self.create_unique_id();
             return nb::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
           })
      .def("connect",
           [](mscclpp::ep::Buffer& self, nb::handle data) {
             std::string s = bytesToString(data);
             mscclpp::UniqueId uid;
             if (s.size() != uid.size()) {
               throw std::runtime_error("mscclpp_ep_cpp.ExpertParallelRuntime.connect: UniqueId size mismatch");
             }
             std::memcpy(uid.data(), s.data(), s.size());
             self.connect(uid);
           })
      .def("sync",
           [](mscclpp::ep::Buffer& self, const std::vector<int>& deviceIds, nb::sequence allGatheredHandles,
              nb::handle rootUniqueId) {
             self.sync(deviceIds, optionalBytesList(allGatheredHandles), optionalBytes(rootUniqueId));
           })
      .def(
          "get_dispatch_layout",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& topkIdx, int numExperts,
             std::optional<mscclpp::ep::EventHandle> previousEvent, bool async, bool allocateOnCommStream) {
            return self.get_dispatch_layout(topkIdx, numExperts, previousEvent, async, allocateOnCommStream);
          },
          nb::arg("topk_idx"), nb::arg("num_experts"), nb::arg("previous_event").none() = nb::none(),
          nb::arg("async") = false, nb::arg("allocate_on_comm_stream") = false)
      .def(
          "intranode_dispatch",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, nb::handle xScales, nb::handle topkIdx,
             nb::handle topkWeights, nb::handle numTokensPerRank, const torch::Tensor& isTokenInRank,
             nb::handle numTokensPerExpert, int cachedNumRecvTokens, nb::handle cachedRankPrefixMatrix,
             nb::handle cachedChannelPrefixMatrix, int expertAlignment, const mscclpp::ep::Config& config,
             std::optional<mscclpp::ep::EventHandle> previousEvent, bool async, bool allocateOnCommStream) {
            return self.intranode_dispatch(
                x, optionalTensor(xScales), optionalTensor(topkIdx), optionalTensor(topkWeights),
                optionalTensor(numTokensPerRank), isTokenInRank, optionalTensor(numTokensPerExpert),
                cachedNumRecvTokens, optionalTensor(cachedRankPrefixMatrix), optionalTensor(cachedChannelPrefixMatrix),
                expertAlignment, config, previousEvent, async, allocateOnCommStream);
          },
          nb::arg("x"), nb::arg("x_scales").none(), nb::arg("topk_idx").none(), nb::arg("topk_weights").none(),
          nb::arg("num_tokens_per_rank").none(), nb::arg("is_token_in_rank"), nb::arg("num_tokens_per_expert").none(),
          nb::arg("cached_num_recv_tokens"), nb::arg("cached_rank_prefix_matrix").none(),
          nb::arg("cached_channel_prefix_matrix").none(), nb::arg("expert_alignment"), nb::arg("config"),
          nb::arg("previous_event").none(), nb::arg("async"), nb::arg("allocate_on_comm_stream"))
      .def(
          "intranode_combine",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, nb::handle topkWeights, const torch::Tensor& srcIdx,
             const torch::Tensor& rankPrefixMatrix, const torch::Tensor& channelPrefixMatrix,
             const torch::Tensor& sendHead, const mscclpp::ep::Config& config,
             std::optional<mscclpp::ep::EventHandle> previousEvent, bool async, bool allocateOnCommStream) {
            return self.intranode_combine(x, optionalTensor(topkWeights), srcIdx, rankPrefixMatrix, channelPrefixMatrix,
                                          sendHead, config, previousEvent, async, allocateOnCommStream);
          },
          nb::arg("x"), nb::arg("topk_weights").none(), nb::arg("src_idx"), nb::arg("rank_prefix_matrix"),
          nb::arg("channel_prefix_matrix"), nb::arg("send_head"), nb::arg("config"), nb::arg("previous_event").none(),
          nb::arg("async"), nb::arg("allocate_on_comm_stream"))
      .def(
          "internode_dispatch",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, nb::handle xScales, nb::handle topkIdx,
             nb::handle topkWeights, nb::handle numTokensPerRank, nb::handle numTokensPerRdmaRank,
             const torch::Tensor& isTokenInRank, nb::handle numTokensPerExpert, int cachedNumRecvTokens,
             int cachedNumRdmaRecvTokens, nb::handle cachedRdmaChannelPrefixMatrix,
             nb::handle cachedRecvRdmaRankPrefixSum, nb::handle cachedGblChannelPrefixMatrix,
             nb::handle cachedRecvGblRankPrefixSum, int expertAlignment, const mscclpp::ep::Config& config,
             std::optional<mscclpp::ep::EventHandle> previousEvent, bool async, bool allocateOnCommStream) {
            return self.internode_dispatch(
                x, optionalTensor(xScales), optionalTensor(topkIdx), optionalTensor(topkWeights),
                optionalTensor(numTokensPerRank), optionalTensor(numTokensPerRdmaRank), isTokenInRank,
                optionalTensor(numTokensPerExpert), cachedNumRecvTokens, cachedNumRdmaRecvTokens,
                optionalTensor(cachedRdmaChannelPrefixMatrix), optionalTensor(cachedRecvRdmaRankPrefixSum),
                optionalTensor(cachedGblChannelPrefixMatrix), optionalTensor(cachedRecvGblRankPrefixSum),
                expertAlignment, config, previousEvent, async, allocateOnCommStream);
          },
          nb::arg("x"), nb::arg("x_scales").none(), nb::arg("topk_idx").none(), nb::arg("topk_weights").none(),
          nb::arg("num_tokens_per_rank").none(), nb::arg("num_tokens_per_rdma_rank").none(),
          nb::arg("is_token_in_rank"), nb::arg("num_tokens_per_expert").none(), nb::arg("cached_num_recv_tokens"),
          nb::arg("cached_num_rdma_recv_tokens"), nb::arg("cached_rdma_channel_prefix_matrix").none(),
          nb::arg("cached_recv_rdma_rank_prefix_sum").none(), nb::arg("cached_gbl_channel_prefix_matrix").none(),
          nb::arg("cached_recv_gbl_rank_prefix_sum").none(), nb::arg("expert_alignment"), nb::arg("config"),
          nb::arg("previous_event").none(), nb::arg("async"), nb::arg("allocate_on_comm_stream"))
      .def(
          "internode_combine",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, nb::handle topkWeights, const torch::Tensor& srcMeta,
             const torch::Tensor& isCombinedTokenInRank, const torch::Tensor& rdmaChannelPrefixMatrix,
             const torch::Tensor& rdmaRankPrefixSum, const torch::Tensor& gblChannelPrefixMatrix,
             const torch::Tensor& combinedRdmaHead, const torch::Tensor& combinedNvlHead,
             const mscclpp::ep::Config& config, std::optional<mscclpp::ep::EventHandle> previousEvent, bool async,
             bool allocateOnCommStream) {
            return self.internode_combine(x, optionalTensor(topkWeights), srcMeta, isCombinedTokenInRank,
                                          rdmaChannelPrefixMatrix, rdmaRankPrefixSum, gblChannelPrefixMatrix,
                                          combinedRdmaHead, combinedNvlHead, config, previousEvent, async,
                                          allocateOnCommStream);
          },
          nb::arg("x"), nb::arg("topk_weights").none(), nb::arg("src_meta"), nb::arg("is_combined_token_in_rank"),
          nb::arg("rdma_channel_prefix_matrix"), nb::arg("rdma_rank_prefix_sum"), nb::arg("gbl_channel_prefix_matrix"),
          nb::arg("combined_rdma_head"), nb::arg("combined_nvl_head"), nb::arg("config"),
          nb::arg("previous_event").none(), nb::arg("async"), nb::arg("allocate_on_comm_stream"))
      .def("clean_low_latency_buffer", &mscclpp::ep::Buffer::clean_low_latency_buffer)
      .def(
          "low_latency_dispatch",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, const torch::Tensor& topkIdx,
             int numMaxDispatchTokensPerRank, int numExperts, bool useFp8, bool async, bool returnRecvHook,
             nb::handle outPackedRecvX, nb::handle outPackedRecvXScales, nb::handle outPackedRecvSrcInfo,
             nb::handle outPackedRecvLayoutRange, nb::handle outPackedRecvCount) {
            return self.low_latency_dispatch(x, topkIdx, numMaxDispatchTokensPerRank, numExperts, useFp8, async,
                                             returnRecvHook, optionalTensor(outPackedRecvX),
                                             optionalTensor(outPackedRecvXScales), optionalTensor(outPackedRecvSrcInfo),
                                             optionalTensor(outPackedRecvLayoutRange),
                                             optionalTensor(outPackedRecvCount));
          },
          nb::arg("x"), nb::arg("topk_idx"), nb::arg("num_max_dispatch_tokens_per_rank"), nb::arg("num_experts"),
          nb::arg("use_fp8"), nb::arg("async"), nb::arg("return_recv_hook"),
          nb::arg("out_packed_recv_x").none() = nb::none(), nb::arg("out_packed_recv_x_scales").none() = nb::none(),
          nb::arg("out_packed_recv_src_info").none() = nb::none(),
          nb::arg("out_packed_recv_layout_range").none() = nb::none(),
          nb::arg("out_packed_recv_count").none() = nb::none())
      .def(
          "low_latency_combine",
          [](mscclpp::ep::Buffer& self, const torch::Tensor& x, nb::handle xScales, const torch::Tensor& topkIdx,
             const torch::Tensor& topkWeights, const torch::Tensor& srcInfo, const torch::Tensor& layoutRange,
             int numMaxDispatchTokensPerRank, int numExperts, bool zeroCopy, bool async, bool returnRecvHook,
             nb::handle out) {
            return self.low_latency_combine(x, optionalTensor(xScales), topkIdx, topkWeights, srcInfo, layoutRange,
                                            numMaxDispatchTokensPerRank, numExperts, zeroCopy, async, returnRecvHook,
                                            optionalTensor(out));
          },
          nb::arg("x"), nb::arg("x_scales").none(), nb::arg("topk_idx"), nb::arg("topk_weights"), nb::arg("src_info"),
          nb::arg("layout_range"), nb::arg("num_max_dispatch_tokens_per_rank"), nb::arg("num_experts"),
          nb::arg("zero_copy"), nb::arg("async"), nb::arg("return_recv_hook"), nb::arg("out").none() = nb::none())
      .def("get_next_low_latency_combine_buffer", &mscclpp::ep::Buffer::get_next_low_latency_combine_buffer);
}
