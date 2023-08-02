// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/fifo.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_fifo(nb::module_& m) {
  nb::class_<ProxyTrigger>(m, "ProxyTrigger").def_rw("fst", &ProxyTrigger::fst).def_rw("snd", &ProxyTrigger::snd);

  nb::class_<FifoDeviceHandle>(m, "FifoDeviceHandle")
      .def_rw("triggers", &FifoDeviceHandle::triggers)
      .def_rw("tail_replica", &FifoDeviceHandle::tailReplica)
      .def_rw("head", &FifoDeviceHandle::head);

  nb::class_<Fifo>(m, "Fifo")
      .def(nb::init<>())
      .def("poll", &Fifo::poll, nb::arg("trigger"))
      .def("pop", &Fifo::pop)
      .def("flush_tail", &Fifo::flushTail, nb::arg("sync") = false)
      .def("device_fifo", &Fifo::deviceHandle);
}
