// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_utils(nb::module_& m) { m.def("get_host_name", &getHostName, nb::arg("maxlen"), nb::arg("delim")); }
