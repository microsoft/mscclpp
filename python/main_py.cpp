// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/core.hpp>

namespace nb = nanobind;

extern void register_error(nb::module_& m);
extern void register_proxy_channel(nb::module_& m);
extern void register_core(nb::module_& m);

NB_MODULE(mscclpp, m) {
  register_error(m);
  register_proxy_channel(m);
  register_core(m);
}
