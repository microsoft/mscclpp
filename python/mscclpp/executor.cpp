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
      .value("float32", DataType::FLOAT32);

  nb::class_<ExecutionPlan>(m, "ExecutionPlan").def(nb::init<std::string>(), nb::arg("planPath"));

  nb::class_<Executor>(m, "Executor")
      .def(nb::init<std::shared_ptr<Communicator>, int>(), nb::arg("comm"), nb::arg("nranksPerNode"))
      .def(
          "execute",
          [](Executor* self, int rank, uintptr_t sendbuff, uintptr_t recvBuff, size_t sendBuffSize, size_t recvBuffSize,
             DataType dataType, int nthreads, const ExecutionPlan& plan, uintptr_t stream) {
            self->execute(rank, reinterpret_cast<void*>(sendbuff), reinterpret_cast<void*>(recvBuff), sendBuffSize,
                          recvBuffSize, dataType, nthreads, plan, (cudaStream_t)stream);
          },
          nb::arg("rank"), nb::arg("sendbuff"), nb::arg("recvBuff"), nb::arg("sendBuffSize"), nb::arg("recvBuffSize"),
          nb::arg("dataType"), nb::arg("nthreads"), nb::arg("plan"), nb::arg("stream"));
}
