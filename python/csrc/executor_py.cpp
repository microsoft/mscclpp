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
  nb::enum_<DataType>(m, "DataType")
      .value("int32", DataType::INT32)
      .value("uint32", DataType::UINT32)
      .value("float16", DataType::FLOAT16)
      .value("float32", DataType::FLOAT32)
      .value("bfloat16", DataType::BFLOAT16);

  nb::enum_<PacketType>(m, "PacketType").value("LL8", PacketType::LL8).value("LL16", PacketType::LL16);

  nb::class_<ExecutionRequest>(m, "ExecutionRequest")
      .def_ro("world_size", &ExecutionRequest::worldSize)
      .def_ro("n_ranks_per_node", &ExecutionRequest::nRanksPerNode)
      .def_prop_ro(
          "input_buffer",
          [](const ExecutionRequest& self) -> uintptr_t { return reinterpret_cast<uintptr_t>(self.inputBuffer); })
      .def_prop_ro(
          "output_buffer",
          [](const ExecutionRequest& self) -> uintptr_t { return reinterpret_cast<uintptr_t>(self.outputBuffer); })
      .def_ro("message_size", &ExecutionRequest::messageSize)
      .def_prop_ro("collective", [](ExecutionRequest& self) -> const std::string& { return self.collective; })
      .def_prop_ro("hints", [](ExecutionRequest& self) { return self.hints; });

  nb::class_<ExecutionPlanHandle>(m, "ExecutionPlanHandle")
      .def_ro("id", &ExecutionPlanHandle::id)
      .def_ro("constraint", &ExecutionPlanHandle::constraint)
      .def_ro("plan", &ExecutionPlanHandle::plan)
      .def_ro("tags", &ExecutionPlanHandle::tags)
      .def_static("create", &ExecutionPlanHandle::create, nb::arg("id"), nb::arg("world_size"),
                  nb::arg("nranks_per_node"), nb::arg("plan"),
                  nb::arg("tags") = std::unordered_map<std::string, uint64_t>{});

  nb::class_<ExecutionPlanHandle::Constraint>(m, "ExecutionPlanConstraint")
      .def_ro("world_size", &ExecutionPlanHandle::Constraint::worldSize)
      .def_ro("n_ranks_per_node", &ExecutionPlanHandle::Constraint::nRanksPerNode);

  nb::class_<ExecutionPlanRegistry>(m, "ExecutionPlanRegistry")
      .def_static("get_instance", &ExecutionPlanRegistry::getInstance)
      .def("register_plan", &ExecutionPlanRegistry::registerPlan, nb::arg("planHandle"))
      .def("get_plans", &ExecutionPlanRegistry::getPlans, nb::arg("collective"))
      .def("get", &ExecutionPlanRegistry::get, nb::arg("id"))
      .def("set_selector", &ExecutionPlanRegistry::setSelector, nb::arg("selector"))
      .def("set_default_selector", &ExecutionPlanRegistry::setDefaultSelector, nb::arg("selector"))
      .def("clear", &ExecutionPlanRegistry::clear);

  nb::class_<ExecutionPlan>(m, "ExecutionPlan")
      .def(nb::init<const std::string&, int>(), nb::arg("planPath"), nb::arg("rank"))
      .def_prop_ro("name", [](const ExecutionPlan& self) -> std::string { return self.name(); })
      .def_prop_ro("collective", [](const ExecutionPlan& self) -> std::string { return self.collective(); })
      .def_prop_ro("min_message_size", [](const ExecutionPlan& self) -> size_t { return self.minMessageSize(); })
      .def_prop_ro("max_message_size", [](const ExecutionPlan& self) -> size_t { return self.maxMessageSize(); });

  nb::class_<Executor>(m, "Executor")
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
