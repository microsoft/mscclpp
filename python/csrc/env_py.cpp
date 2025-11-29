// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/env.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_env(nb::module_& m) {
  nb::class_<Env>(m, "Env")
      .def_ro("debug", &Env::debug)
      .def_ro("debug_subsys", &Env::debugSubsys)
      .def_ro("debug_file", &Env::debugFile)
      .def_ro("hca_devices", &Env::hcaDevices)
      .def_ro("hostid", &Env::hostid)
      .def_ro("socket_family", &Env::socketFamily)
      .def_ro("socket_ifname", &Env::socketIfname)
      .def_ro("comm_id", &Env::commId)
      .def_ro("execution_plan_dir", &Env::executionPlanDir)
      .def_ro("npkit_dump_dir", &Env::npkitDumpDir)
      .def_ro("cuda_ipc_use_default_stream", &Env::cudaIpcUseDefaultStream);

  m.def("env", &env);
}
