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

  nb::class_<SmChannel::DeviceHandle>(smChannel, "DeviceHandle");

  m.def("device_handle", &deviceHandle<SmChannel>, nb::arg("smChannel"));
};
