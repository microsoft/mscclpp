// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/config.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_config(nb::module_& m) {
  nb::class_<Config>(m, "Config")
      .def_static("get_instance", &Config::getInstance, nb::rv_policy::reference)
      .def("get_bootstrap_connection_timeout_config", &Config::getBootstrapConnectionTimeoutConfig)
      .def("set_bootstrap_connection_timeout_config", &Config::setBootstrapConnectionTimeoutConfig);
}
