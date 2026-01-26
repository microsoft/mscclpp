// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/executor.hpp>
#include <mscclpp/gpu.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_executor(nb::module_& m) {
  nb::enum_<PacketType>(m, "CppPacketType").value("LL8", PacketType::LL8).value("LL16", PacketType::LL16);

  nb::class_<ExecutionPlan>(m, "CppExecutionPlan")
      .def(nb::init<const std::string&, int>(), nb::arg("planPath"), nb::arg("rank"))
      .def_prop_ro("name", [](const ExecutionPlan& self) -> std::string { return self.name(); })
      .def_prop_ro("collective", [](const ExecutionPlan& self) -> std::string { return self.collective(); })
      .def_prop_ro("min_message_size", [](const ExecutionPlan& self) -> size_t { return self.minMessageSize(); })
      .def_prop_ro("max_message_size", [](const ExecutionPlan& self) -> size_t { return self.maxMessageSize(); });

  nb::class_<Executor>(m, "CppExecutor")
      .def(nb::init<std::shared_ptr<Communicator>>(), nb::arg("comm"))
      .def(
          "execute",
          [](Executor* self, int rank, uintptr_t sendbuff, uintptr_t recvBuff, size_t sendBuffSize, size_t recvBuffSize,
             DataType dataType, const ExecutionPlan& plan, uintptr_t stream, PacketType packetType) {
            self->execute(rank, reinterpret_cast<void*>(sendbuff), reinterpret_cast<void*>(recvBuff), sendBuffSize,
                          recvBuffSize, dataType, plan, (cudaStream_t)stream, packetType);
          },
          nb::arg("rank"), nb::arg("send_buff"), nb::arg("recv_buff"), nb::arg("send_buff_size"),
          nb::arg("recv_buff_size"), nb::arg("data_type"), nb::arg("plan"), nb::arg("stream"),
          nb::arg("packet_type") = PacketType::LL16);
}
