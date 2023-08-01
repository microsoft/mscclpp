// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/sm_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_sm_channel(nb::module_& m) {
  nb::class_<SmChannel> smChannel(m, "SmChannel");
  smChannel
      .def("__init__",
           [](SmChannel* smChannel, std::shared_ptr<SmDevice2DeviceSemaphore> semaphore, RegisteredMemory dst,
              uintptr_t src) { new (smChannel) SmChannel(semaphore, dst, (void*)src); })
      .def("device_handle", &SmChannel::deviceHandle);

  m.def("device_handle", &deviceHandle<SmChannel>, nb::arg("smChannel"));

  nb::class_<SmChannel::DeviceHandle>(smChannel, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphore_", &SmChannel::DeviceHandle::semaphore_)
      .def_rw("src_", &SmChannel::DeviceHandle::src_)
      .def_rw("dst_", &SmChannel::DeviceHandle::dst_)
      .def_rw("getPacketBuffer_", &SmChannel::DeviceHandle::getPacketBuffer_)
      .def_prop_ro("raw", [](const SmChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

};
