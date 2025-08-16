// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>

#include <mscclpp/fifo.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_fifo(nb::module_& m) {
  nb::class_<ProxyTrigger>(m, "ProxyTrigger")
      .def_prop_rw(
          "fst", [](const ProxyTrigger& self) { return self.fst; },
          [](ProxyTrigger& self, uint64_t v) { self.fst = v; })
      .def_prop_rw(
          "snd", [](const ProxyTrigger& self) { return self.snd; },
          [](ProxyTrigger& self, uint64_t v) { self.snd = v; });

  nb::class_<FifoDeviceHandle>(m, "FifoDeviceHandle")
      .def_rw("triggers", &FifoDeviceHandle::triggers)
      .def_rw("tail", &FifoDeviceHandle::tail)
      .def_rw("head", &FifoDeviceHandle::head)
      .def_rw("size", &FifoDeviceHandle::size)
      .def_prop_ro("raw", [](const FifoDeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<Fifo>(m, "Fifo")
      .def(nb::init<int>(), nb::arg("size") = DEFAULT_FIFO_SIZE)
      .def("poll", &Fifo::poll)
      .def("pop", &Fifo::pop)
      .def("size", &Fifo::size)
      .def("device_handle", &Fifo::deviceHandle);
}
