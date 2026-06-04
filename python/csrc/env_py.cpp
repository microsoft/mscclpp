// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/env.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_env(nb::module_& m) {
  nb::class_<Env>(m, "CppEnv")
      .def_ro("debug", &Env::debug)
      .def_ro("debug_subsys", &Env::debugSubsys)
      .def_ro("debug_file", &Env::debugFile)
      .def_ro("hca_devices", &Env::hcaDevices)
      .def_ro("hostid", &Env::hostid)
      .def_ro("socket_family", &Env::socketFamily)
      .def_ro("socket_ifname", &Env::socketIfname)
      .def_ro("comm_id", &Env::commId)
      .def_ro("ibv_mode", &Env::ibvMode)
      .def_ro("cache_dir", &Env::cacheDir)
      .def_ro("npkit_dump_dir", &Env::npkitDumpDir)
      .def_ro("cuda_ipc_use_default_stream", &Env::cudaIpcUseDefaultStream)
      .def_ro("nccl_shared_lib_path", &Env::ncclSharedLibPath)
      .def_ro("force_nccl_fallback_operation", &Env::forceNcclFallbackOperation)
      .def_ro("nccl_symmetric_memory", &Env::ncclSymmetricMemory)
      .def_ro("force_disable_nvls", &Env::forceDisableNvls)
      .def_ro("force_disable_gdr", &Env::forceDisableGdr)
      .def_ro("ib_gid_index", &Env::ibGidIndex);

  m.def("env", &env);
}
