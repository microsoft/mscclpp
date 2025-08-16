// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/core.hpp>
#include <mscclpp/nvls.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_nvls(nb::module_& m) {
  nb::class_<SwitchChannel>(m, "SwitchChannel")
      .def("get_device_ptr", [](SwitchChannel* self) { return (uintptr_t)self->getDevicePtr(); })
      .def("device_handle", &SwitchChannel::deviceHandle);

  nb::class_<SwitchChannel::DeviceHandle>(m, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("devicePtr", &SwitchChannel::DeviceHandle::devicePtr)
      .def_rw("mcPtr", &SwitchChannel::DeviceHandle::mcPtr)
      .def_rw("size", &SwitchChannel::DeviceHandle::bufferSize)
      .def_prop_ro("raw", [](const SwitchChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<NvlsConnection>(m, "NvlsConnection")
      .def("bind_allocated_memory", &NvlsConnection::bindAllocatedMemory, nb::arg("devicePtr"), nb::arg("size"))
      .def("get_multicast_min_granularity", &NvlsConnection::getMultiCastMinGranularity);

  m.def("connect_nvls_collective", &connectNvlsCollective, nb::arg("communicator"), nb::arg("allRanks"),
        nb::arg("bufferSize"));
}
