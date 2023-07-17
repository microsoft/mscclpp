// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/fifo.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_fifo(nb::module_& m) {
  nb::class_<ProxyTrigger>(m, "ProxyTrigger").def_rw("fst", &ProxyTrigger::fst).def_rw("snd", &ProxyTrigger::snd);

  nb::class_<DeviceProxyFifo>(m, "DeviceProxyFifo")
      .def_rw("triggers", &DeviceProxyFifo::triggers)
      .def_rw("tail_replica", &DeviceProxyFifo::tailReplica)
      .def_rw("head", &DeviceProxyFifo::head);

  nb::class_<HostProxyFifo>(m, "HostProxyFifo")
      .def(nb::init<>())
      .def("poll", &HostProxyFifo::poll, nb::arg("trigger"))
      .def("pop", &HostProxyFifo::pop)
      .def("flush_tail", &HostProxyFifo::flushTail, nb::arg("sync") = false)
      .def("device_fifo", &HostProxyFifo::deviceFifo);
}
