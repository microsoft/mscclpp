// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/core.hpp>
#include <mscclpp/switch_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_nvls(nb::module_& m) {
  nb::class_<SwitchChannel>(m, "CppSwitchChannel")
      .def("get_device_ptr", [](SwitchChannel* self) { return (uintptr_t)self->getDevicePtr(); })
      .def("device_handle", &SwitchChannel::deviceHandle);

  nb::class_<SwitchChannel::DeviceHandle>(m, "CppSwitchChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("device_ptr", &SwitchChannel::DeviceHandle::devicePtr)
      .def_rw("mc_ptr", &SwitchChannel::DeviceHandle::mcPtr)
      .def_rw("size", &SwitchChannel::DeviceHandle::bufferSize)
      .def_prop_ro("raw", [](const SwitchChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<NvlsConnection>(m, "CppNvlsConnection")
      .def("bind_allocated_memory", &NvlsConnection::bindAllocatedMemory, nb::arg("device_ptr"), nb::arg("size"));

  m.def("connect_nvls_collective", &connectNvlsCollective, nb::arg("communicator"), nb::arg("all_ranks"),
        nb::arg("buffer_size"));
}
