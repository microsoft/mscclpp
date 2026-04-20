// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// pybind11 module definition for the MSCCL++ EP extension. Mirrors
// DeepEP's `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` so call sites port
// with minimal changes.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include "buffer.hpp"
#include "config.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MSCCL++ Expert-Parallel (MoE dispatch/combine) extension";

  py::class_<mscclpp::ep::Config>(m, "Config")
      .def(py::init<int, int, int, int, int>(), py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &mscclpp::ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint", &mscclpp::ep::Config::get_rdma_buffer_size_hint);

  m.def("get_low_latency_rdma_size_hint", &mscclpp::ep::get_low_latency_rdma_size_hint);

  py::class_<mscclpp::ep::EventHandle>(m, "EventHandle")
      .def(py::init<>())
      .def("current_stream_wait", &mscclpp::ep::EventHandle::current_stream_wait);

  // NOTE: `mscclpp::UniqueId` is the bootstrap id used for connecting the
  // proxy service. We expose it as an opaque bytes-like object so Python can
  // all-gather it across the user's process group.
  py::class_<mscclpp::UniqueId>(m, "UniqueId")
      .def(py::init<>())
      .def("bytes", [](const mscclpp::UniqueId& self) {
        return py::bytes(reinterpret_cast<const char*>(self.data()), self.size());
      })
      .def_static("from_bytes", [](py::bytes data) {
        auto s = std::string(data);
        mscclpp::UniqueId uid;
        if (s.size() != uid.size()) {
          throw std::runtime_error("mscclpp.ep.UniqueId.from_bytes: size mismatch");
        }
        std::memcpy(uid.data(), s.data(), s.size());
        return uid;
      });

  py::class_<mscclpp::ep::Buffer>(m, "Buffer")
      .def(py::init<int, int, int64_t, int64_t, bool>(), py::arg("rank"), py::arg("num_ranks"),
           py::arg("num_nvl_bytes"), py::arg("num_rdma_bytes"), py::arg("low_latency_mode"))
      .def("is_available", &mscclpp::ep::Buffer::is_available)
      .def("is_internode_available", &mscclpp::ep::Buffer::is_internode_available)
      .def("get_num_rdma_ranks", &mscclpp::ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &mscclpp::ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &mscclpp::ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &mscclpp::ep::Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &mscclpp::ep::Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id", &mscclpp::ep::Buffer::get_local_nvshmem_unique_id)
      .def("get_local_buffer_tensor", &mscclpp::ep::Buffer::get_local_buffer_tensor)
      .def("create_unique_id", &mscclpp::ep::Buffer::create_unique_id)
      .def("connect", &mscclpp::ep::Buffer::connect)
      .def("sync", &mscclpp::ep::Buffer::sync)
      .def("get_dispatch_layout", &mscclpp::ep::Buffer::get_dispatch_layout)
      .def("intranode_dispatch", &mscclpp::ep::Buffer::intranode_dispatch)
      .def("intranode_combine", &mscclpp::ep::Buffer::intranode_combine)
      .def("internode_dispatch", &mscclpp::ep::Buffer::internode_dispatch)
      .def("internode_combine", &mscclpp::ep::Buffer::internode_combine)
      .def("clean_low_latency_buffer", &mscclpp::ep::Buffer::clean_low_latency_buffer)
      .def("low_latency_dispatch", &mscclpp::ep::Buffer::low_latency_dispatch)
      .def("low_latency_combine", &mscclpp::ep::Buffer::low_latency_combine)
      .def("get_next_low_latency_combine_buffer", &mscclpp::ep::Buffer::get_next_low_latency_combine_buffer);
}
