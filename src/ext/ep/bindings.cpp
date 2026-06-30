// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// nanobind module definition for the MSCCL++ EP extension.
//
// Two backends are exposed in one module:
//   - MoERuntime: low-latency (LL) path. Pointer-based API (uintptr_t), no
//     torch at the boundary.
//   - Buffer: high-throughput (HT) DeepEP-style path. Carries `torch::Tensor`;
//     tensors cross the Python boundary as DLPack capsules (ATen toDLPack /
//     fromDLPack), so this module links libtorch (ATen) but never libtorch_python
//     (which would pull pybind11 into the nanobind TU).

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
#include "kernels/api.cuh"
#include "moe_runtime.hpp"

// HT backend (DeepEP-style Buffer). ATen-level torch headers only (no pybind11).
#include <ATen/DLConvertor.h>

#include "ht/buffer.hpp"
#include "ht/config.hpp"

namespace nb = nanobind;

namespace {

nb::bytes stringToBytes(const std::string& data) { return nb::bytes(data.data(), data.size()); }

void* ptr(uintptr_t address) { return reinterpret_cast<void*>(address); }

cudaStream_t stream(uintptr_t address) { return reinterpret_cast<cudaStream_t>(address); }

// ----------------------------------------------------------------------------
// DLPack bridge: torch::Tensor <-> Python, via DLPack capsules.
//
// Python wraps the HT Buffer (buffer.py): it passes inputs as
// `torch.utils.dlpack.to_dlpack(t)` capsules and rebuilds outputs with
// `torch.from_dlpack(capsule)`. The capsule ownership follows the standard
// DLPack protocol (capsule named "dltensor"; the consumer renames it to
// "used_dltensor" once it takes ownership).
// ----------------------------------------------------------------------------

void dlpackCapsuleDestructor(PyObject* capsule) {
  // Only free if the receiver never consumed it (torch renames to
  // "used_dltensor" when it takes ownership of the managed tensor).
  if (PyCapsule_IsValid(capsule, "used_dltensor")) return;
  auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
  if (mt != nullptr && mt->deleter != nullptr) {
    mt->deleter(mt);
  } else {
    PyErr_Clear();
  }
}

// torch::Tensor -> owning Python DLPack capsule named "dltensor".
nb::object tensorToCapsule(const torch::Tensor& t) {
  DLManagedTensor* mt = at::toDLPack(t);
  PyObject* cap = PyCapsule_New(mt, "dltensor", dlpackCapsuleDestructor);
  if (cap == nullptr) {
    if (mt->deleter != nullptr) mt->deleter(mt);
    throw std::runtime_error("mscclpp_ep: failed to create DLPack capsule");
  }
  return nb::steal(cap);
}

nb::object optTensorToCapsule(const std::optional<torch::Tensor>& t) {
  if (!t.has_value()) return nb::none();
  return tensorToCapsule(*t);
}

// Python DLPack capsule -> torch::Tensor (consumes the capsule).
torch::Tensor capsuleToTensor(nb::handle obj) {
  PyObject* cap = obj.ptr();
  if (!PyCapsule_IsValid(cap, "dltensor")) {
    throw std::runtime_error("mscclpp_ep: expected a DLPack capsule named 'dltensor'");
  }
  auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
  PyCapsule_SetName(cap, "used_dltensor");
  return at::fromDLPack(mt);
}

std::optional<torch::Tensor> objToOptTensor(nb::handle obj) {
  if (obj.is_none()) return std::nullopt;
  return capsuleToTensor(obj);
}

nb::object eventToObj(std::optional<mscclpp::ep::EventHandle>& ev) {
  if (!ev.has_value()) return nb::none();
  return nb::cast(*ev);
}

std::optional<mscclpp::ep::EventHandle> objToOptEvent(nb::handle o) {
  if (o.is_none()) return std::nullopt;
  return nb::cast<mscclpp::ep::EventHandle>(o);
}

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
             int numExperts, bool requiresQuantization, mscclpp::ep::DispatchLayout outputLayout, uintptr_t streamPtr) {
            self.dispatch(
                ptr(outputPtr), reinterpret_cast<float*>(ptr(outputScalesPtr)),
                reinterpret_cast<int*>(ptr(outputSrcInfoPtr)), reinterpret_cast<int64_t*>(ptr(outputLayoutRangePtr)),
                reinterpret_cast<int*>(ptr(outputCountPtr)), ptr(inputPtr), reinterpret_cast<int64_t*>(ptr(topkIdxPtr)),
                numTokens, hidden, numTopk, numMaxDispatchTokensPerRank, numExperts, requiresQuantization, outputLayout,
                stream(streamPtr));
          },
          nb::arg("input_ptr"), nb::arg("topk_idx_ptr"), nb::arg("output_ptr"), nb::arg("output_scales_ptr"),
          nb::arg("output_src_info_ptr"), nb::arg("output_layout_range_ptr"), nb::arg("output_count_ptr"),
          nb::arg("num_tokens"), nb::arg("hidden"), nb::arg("num_topk"), nb::arg("num_max_dispatch_tokens_per_rank"),
          nb::arg("num_experts"), nb::arg("requires_quantization"), nb::arg("output_layout"), nb::arg("stream_ptr"))
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
  // High-throughput (HT) DeepEP-style backend: Config, EventHandle, Buffer.
  // Tensors cross the boundary as DLPack capsules (see helpers above).
  // ==========================================================================

  nb::class_<mscclpp::ep::Config>(m, "Config")
      .def(nb::init<int, int, int, int, int>(), nb::arg("num_sms") = 20, nb::arg("num_max_nvl_chunked_send_tokens") = 6,
           nb::arg("num_max_nvl_chunked_recv_tokens") = 256, nb::arg("num_max_rdma_chunked_send_tokens") = 6,
           nb::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &mscclpp::ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint", &mscclpp::ep::Config::get_rdma_buffer_size_hint);

  nb::class_<mscclpp::ep::EventHandle>(m, "EventHandle")
      .def(nb::init<>())
      .def("current_stream_wait", &mscclpp::ep::EventHandle::current_stream_wait);

  nb::class_<mscclpp::ep::Buffer>(m, "ExpertParallelRuntime")
      .def(nb::init<int, int, int64_t, int64_t, bool>(), nb::arg("rank").none(), nb::arg("num_ranks").none(),
           nb::arg("num_nvl_bytes").none(), nb::arg("num_rdma_bytes").none(), nb::arg("low_latency_mode").none())
      .def("is_available", &mscclpp::ep::Buffer::is_available)
      .def("is_internode_available", &mscclpp::ep::Buffer::is_internode_available)
      .def("get_num_rdma_ranks", &mscclpp::ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &mscclpp::ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &mscclpp::ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &mscclpp::ep::Buffer::get_local_device_id)
      .def("get_local_ipc_handle",
           [](const mscclpp::ep::Buffer& self) { return stringToBytes(self.get_local_ipc_handle()); })
      .def("create_unique_id",
           [](const mscclpp::ep::Buffer& self) {
             auto uid = self.create_unique_id();
             return nb::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
           })
      .def("connect",
           [](mscclpp::ep::Buffer& self, nb::object data) {
             std::string s = bytesToString(data);
             mscclpp::UniqueId uid;
             if (s.size() != uid.size()) {
               throw std::runtime_error("mscclpp_ep_cpp.Buffer.connect: UniqueId size mismatch");
             }
             std::memcpy(uid.data(), s.data(), s.size());
             self.connect(uid);
           })
      .def(
          "sync",
          [](mscclpp::ep::Buffer& self, const std::vector<int>& device_ids, nb::sequence all_gathered_handles,
             nb::object root_unique_id_opt) {
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
          nb::arg("device_ids").none(), nb::arg("all_gathered_handles").none(), nb::arg("root_unique_id").none())
      .def(
          "get_dispatch_layout",
          [](mscclpp::ep::Buffer& self, nb::object topk_idx, int num_experts, nb::object previous_event,
             bool async_finish, bool allocate_on_comm_stream) {
            std::optional<mscclpp::ep::EventHandle> prev = objToOptEvent(previous_event);
            auto r = self.get_dispatch_layout(capsuleToTensor(topk_idx), num_experts, prev, async_finish,
                                              allocate_on_comm_stream);
            return nb::make_tuple(tensorToCapsule(std::get<0>(r)), optTensorToCapsule(std::get<1>(r)),
                                  tensorToCapsule(std::get<2>(r)), tensorToCapsule(std::get<3>(r)),
                                  eventToObj(std::get<4>(r)));
          },
          nb::arg("topk_idx").none(), nb::arg("num_experts").none(), nb::arg("previous_event").none(),
          nb::arg("async_finish").none(), nb::arg("allocate_on_comm_stream").none())
      .def(
          "intranode_dispatch",
          [](mscclpp::ep::Buffer& self, nb::object x, nb::object x_scales, nb::object topk_idx, nb::object topk_weights,
             nb::object num_tokens_per_rank, nb::object is_token_in_rank, nb::object num_tokens_per_expert,
             int cached_num_recv_tokens, nb::object cached_rank_prefix_matrix, nb::object cached_channel_prefix_matrix,
             int expert_alignment, const mscclpp::ep::Config& config, nb::object previous_event, bool async_finish,
             bool allocate_on_comm_stream) {
            std::optional<mscclpp::ep::EventHandle> prev = objToOptEvent(previous_event);
            auto r = self.intranode_dispatch(capsuleToTensor(x), objToOptTensor(x_scales), objToOptTensor(topk_idx),
                                             objToOptTensor(topk_weights), objToOptTensor(num_tokens_per_rank),
                                             capsuleToTensor(is_token_in_rank), objToOptTensor(num_tokens_per_expert),
                                             cached_num_recv_tokens, objToOptTensor(cached_rank_prefix_matrix),
                                             objToOptTensor(cached_channel_prefix_matrix), expert_alignment, config,
                                             prev, async_finish, allocate_on_comm_stream);
            return nb::make_tuple(
                tensorToCapsule(std::get<0>(r)), optTensorToCapsule(std::get<1>(r)), optTensorToCapsule(std::get<2>(r)),
                optTensorToCapsule(std::get<3>(r)), nb::cast(std::get<4>(r)), tensorToCapsule(std::get<5>(r)),
                tensorToCapsule(std::get<6>(r)), tensorToCapsule(std::get<7>(r)), tensorToCapsule(std::get<8>(r)),
                tensorToCapsule(std::get<9>(r)), eventToObj(std::get<10>(r)));
          },
          nb::arg("x").none(), nb::arg("x_scales").none(), nb::arg("topk_idx").none(), nb::arg("topk_weights").none(),
          nb::arg("num_tokens_per_rank").none(), nb::arg("is_token_in_rank").none(),
          nb::arg("num_tokens_per_expert").none(), nb::arg("cached_num_recv_tokens").none(),
          nb::arg("cached_rank_prefix_matrix").none(), nb::arg("cached_channel_prefix_matrix").none(),
          nb::arg("expert_alignment").none(), nb::arg("config").none(), nb::arg("previous_event").none(),
          nb::arg("async_finish").none(), nb::arg("allocate_on_comm_stream").none())
      .def(
          "intranode_combine",
          [](mscclpp::ep::Buffer& self, nb::object x, nb::object topk_weights, nb::object src_idx,
             nb::object rank_prefix_matrix, nb::object channel_prefix_matrix, nb::object send_head,
             const mscclpp::ep::Config& config, nb::object previous_event, bool async_finish,
             bool allocate_on_comm_stream) {
            std::optional<mscclpp::ep::EventHandle> prev = objToOptEvent(previous_event);
            auto r =
                self.intranode_combine(capsuleToTensor(x), objToOptTensor(topk_weights), capsuleToTensor(src_idx),
                                       capsuleToTensor(rank_prefix_matrix), capsuleToTensor(channel_prefix_matrix),
                                       capsuleToTensor(send_head), config, prev, async_finish, allocate_on_comm_stream);
            return nb::make_tuple(tensorToCapsule(std::get<0>(r)), optTensorToCapsule(std::get<1>(r)),
                                  eventToObj(std::get<2>(r)));
          },
          nb::arg("x").none(), nb::arg("topk_weights").none(), nb::arg("src_idx").none(),
          nb::arg("rank_prefix_matrix").none(), nb::arg("channel_prefix_matrix").none(), nb::arg("send_head").none(),
          nb::arg("config").none(), nb::arg("previous_event").none(), nb::arg("async_finish").none(),
          nb::arg("allocate_on_comm_stream").none())
      .def(
          "internode_dispatch",
          [](mscclpp::ep::Buffer& self, nb::object x, nb::object x_scales, nb::object topk_idx, nb::object topk_weights,
             nb::object num_tokens_per_rank, nb::object num_tokens_per_rdma_rank, nb::object is_token_in_rank,
             nb::object num_tokens_per_expert, int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
             nb::object cached_rdma_channel_prefix_matrix, nb::object cached_recv_rdma_rank_prefix_sum,
             nb::object cached_gbl_channel_prefix_matrix, nb::object cached_recv_gbl_rank_prefix_sum,
             int expert_alignment, const mscclpp::ep::Config& config, nb::object previous_event, bool async_finish,
             bool allocate_on_comm_stream) {
            std::optional<mscclpp::ep::EventHandle> prev = objToOptEvent(previous_event);
            auto r = self.internode_dispatch(
                capsuleToTensor(x), objToOptTensor(x_scales), objToOptTensor(topk_idx), objToOptTensor(topk_weights),
                objToOptTensor(num_tokens_per_rank), objToOptTensor(num_tokens_per_rdma_rank),
                capsuleToTensor(is_token_in_rank), objToOptTensor(num_tokens_per_expert), cached_num_recv_tokens,
                cached_num_rdma_recv_tokens, objToOptTensor(cached_rdma_channel_prefix_matrix),
                objToOptTensor(cached_recv_rdma_rank_prefix_sum), objToOptTensor(cached_gbl_channel_prefix_matrix),
                objToOptTensor(cached_recv_gbl_rank_prefix_sum), expert_alignment, config, prev, async_finish,
                allocate_on_comm_stream);
            return nb::make_tuple(
                tensorToCapsule(std::get<0>(r)), optTensorToCapsule(std::get<1>(r)), optTensorToCapsule(std::get<2>(r)),
                optTensorToCapsule(std::get<3>(r)), nb::cast(std::get<4>(r)), tensorToCapsule(std::get<5>(r)),
                tensorToCapsule(std::get<6>(r)), optTensorToCapsule(std::get<7>(r)), tensorToCapsule(std::get<8>(r)),
                optTensorToCapsule(std::get<9>(r)), tensorToCapsule(std::get<10>(r)),
                optTensorToCapsule(std::get<11>(r)), optTensorToCapsule(std::get<12>(r)),
                optTensorToCapsule(std::get<13>(r)), eventToObj(std::get<14>(r)));
          },
          nb::arg("x").none(), nb::arg("x_scales").none(), nb::arg("topk_idx").none(), nb::arg("topk_weights").none(),
          nb::arg("num_tokens_per_rank").none(), nb::arg("num_tokens_per_rdma_rank").none(),
          nb::arg("is_token_in_rank").none(), nb::arg("num_tokens_per_expert").none(),
          nb::arg("cached_num_recv_tokens").none(), nb::arg("cached_num_rdma_recv_tokens").none(),
          nb::arg("cached_rdma_channel_prefix_matrix").none(), nb::arg("cached_recv_rdma_rank_prefix_sum").none(),
          nb::arg("cached_gbl_channel_prefix_matrix").none(), nb::arg("cached_recv_gbl_rank_prefix_sum").none(),
          nb::arg("expert_alignment").none(), nb::arg("config").none(), nb::arg("previous_event").none(),
          nb::arg("async_finish").none(), nb::arg("allocate_on_comm_stream").none())
      .def(
          "internode_combine",
          [](mscclpp::ep::Buffer& self, nb::object x, nb::object topk_weights, nb::object src_meta,
             nb::object is_combined_token_in_rank, nb::object rdma_channel_prefix_matrix,
             nb::object rdma_rank_prefix_sum, nb::object gbl_channel_prefix_matrix, nb::object combined_rdma_head,
             nb::object combined_nvl_head, const mscclpp::ep::Config& config, nb::object previous_event,
             bool async_finish, bool allocate_on_comm_stream) {
            std::optional<mscclpp::ep::EventHandle> prev = objToOptEvent(previous_event);
            auto r = self.internode_combine(
                capsuleToTensor(x), objToOptTensor(topk_weights), capsuleToTensor(src_meta),
                capsuleToTensor(is_combined_token_in_rank), capsuleToTensor(rdma_channel_prefix_matrix),
                capsuleToTensor(rdma_rank_prefix_sum), capsuleToTensor(gbl_channel_prefix_matrix),
                capsuleToTensor(combined_rdma_head), capsuleToTensor(combined_nvl_head), config, prev, async_finish,
                allocate_on_comm_stream);
            return nb::make_tuple(tensorToCapsule(std::get<0>(r)), optTensorToCapsule(std::get<1>(r)),
                                  eventToObj(std::get<2>(r)));
          },
          nb::arg("x").none(), nb::arg("topk_weights").none(), nb::arg("src_meta").none(),
          nb::arg("is_combined_token_in_rank").none(), nb::arg("rdma_channel_prefix_matrix").none(),
          nb::arg("rdma_rank_prefix_sum").none(), nb::arg("gbl_channel_prefix_matrix").none(),
          nb::arg("combined_rdma_head").none(), nb::arg("combined_nvl_head").none(), nb::arg("config").none(),
          nb::arg("previous_event").none(), nb::arg("async_finish").none(), nb::arg("allocate_on_comm_stream").none());
}
