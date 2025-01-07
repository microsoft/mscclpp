// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/executor.hpp>
#include <mscclpp/gpu.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_executor(nb::module_& m) {
  nb::enum_<DataType>(m, "DataType")
      .value("int32", DataType::INT32)
      .value("uint32", DataType::UINT32)
      .value("float16", DataType::FLOAT16)
      .value("float32", DataType::FLOAT32)
      .value("bfloat16", DataType::BFLOAT16);

  nb::enum_<PacketType>(m, "PacketType").value("LL8", PacketType::LL8).value("LL16", PacketType::LL16);

  nb::class_<ExecutionPlan>(m, "ExecutionPlan")
      .def(nb::init<const std::string>(), nb::arg("planPath"))
      .def("name", &ExecutionPlan::name)
      .def("collective", &ExecutionPlan::collective)
      .def("min_message_size", &ExecutionPlan::minMessageSize)
      .def("max_message_size", &ExecutionPlan::maxMessageSize);

  nb::class_<Executor>(m, "Executor")
      .def(nb::init<std::shared_ptr<Communicator>>(), nb::arg("comm"))
      .def(
          "execute",
          [](Executor* self, int rank, uintptr_t sendbuff, uintptr_t recvBuff, size_t sendBuffSize, size_t recvBuffSize,
             DataType dataType, const ExecutionPlan& plan, uintptr_t stream, PacketType packetType) {
            self->execute(rank, reinterpret_cast<void*>(sendbuff), reinterpret_cast<void*>(recvBuff), sendBuffSize,
                          recvBuffSize, dataType, plan, (cudaStream_t)stream, packetType);
          },
          nb::arg("rank"), nb::arg("sendbuff"), nb::arg("recvBuff"), nb::arg("sendBuffSize"), nb::arg("recvBuffSize"),
          nb::arg("dataType"), nb::arg("plan"), nb::arg("stream"), nb::arg("packetType") = PacketType::LL16);
}
