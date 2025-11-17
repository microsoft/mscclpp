// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <mscclpp/semaphore.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_semaphore(nb::module_& m) {
  nb::class_<Host2DeviceSemaphore> host2DeviceSemaphore(m, "Host2DeviceSemaphore");
  host2DeviceSemaphore.def(nb::init<const Semaphore&>(), nb::arg("semaphore"))
      .def(nb::init<Communicator&, const Connection&>(), nb::arg("communicator"), nb::arg("connection"))
      .def("connection", &Host2DeviceSemaphore::connection)
      .def("signal", &Host2DeviceSemaphore::signal)
      .def("device_handle", &Host2DeviceSemaphore::deviceHandle);

  nb::class_<Host2DeviceSemaphore::DeviceHandle>(host2DeviceSemaphore, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("inbound_token", &Host2DeviceSemaphore::DeviceHandle::inboundToken)
      .def_rw("expected_inbound_token", &Host2DeviceSemaphore::DeviceHandle::expectedInboundToken)
      .def_prop_ro("raw", [](const Host2DeviceSemaphore::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<Host2HostSemaphore>(m, "Host2HostSemaphore")
      .def(nb::init<const Semaphore&>(), nb::arg("semaphore"))
      .def(nb::init<Communicator&, const Connection&>(), nb::arg("communicator"), nb::arg("connection"))
      .def("connection", &Host2HostSemaphore::connection)
      .def("signal", &Host2HostSemaphore::signal)
      .def("poll", &Host2HostSemaphore::poll)
      .def("wait", &Host2HostSemaphore::wait, nb::call_guard<nb::gil_scoped_release>(),
           nb::arg("max_spin_count") = 10000000);

  nb::class_<MemoryDevice2DeviceSemaphore> memoryDevice2DeviceSemaphore(m, "MemoryDevice2DeviceSemaphore");
  memoryDevice2DeviceSemaphore.def(nb::init<const Semaphore&>(), nb::arg("semaphore"))
      .def(nb::init<Communicator&, const Connection&>(), nb::arg("communicator"), nb::arg("connection"))
      .def("connection", &MemoryDevice2DeviceSemaphore::connection)
      .def("device_handle", &MemoryDevice2DeviceSemaphore::deviceHandle);

  nb::class_<MemoryDevice2DeviceSemaphore::DeviceHandle>(memoryDevice2DeviceSemaphore, "DeviceHandle")
      .def(nb::init<>())
      .def_rw("inbound_token", &MemoryDevice2DeviceSemaphore::DeviceHandle::inboundToken)
      .def_rw("outbound_token", &MemoryDevice2DeviceSemaphore::DeviceHandle::outboundToken)
      .def_rw("remote_inbound_token", &MemoryDevice2DeviceSemaphore::DeviceHandle::remoteInboundToken)
      .def_rw("expected_inbound_token", &MemoryDevice2DeviceSemaphore::DeviceHandle::expectedInboundToken)
      .def_prop_ro("raw", [](const MemoryDevice2DeviceSemaphore::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
}
