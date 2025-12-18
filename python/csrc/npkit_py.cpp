// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/npkit/npkit.hpp>

namespace nb = nanobind;

void register_npkit(nb::module_ &m) {
  nb::module_ sub_m = m.def_submodule("npkit", "NPKit functions");
  sub_m.def("init", &NpKit::Init);
  sub_m.def("dump", &NpKit::Dump);
  sub_m.def("shutdown", &NpKit::Shutdown);
}
