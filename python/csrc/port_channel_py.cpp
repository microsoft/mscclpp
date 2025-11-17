// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/port_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_port_channel(nb::module_& m) {
  nb::class_<BaseProxyService>(m, "BaseProxyService")
      .def("start_proxy", &BaseProxyService::startProxy, nb::arg("blocking") = false)
      .def("stop_proxy", &BaseProxyService::stopProxy);

  nb::class_<ProxyService, BaseProxyService>(m, "ProxyService")
      .def(nb::init<int>(), nb::arg("fifo_size") = DEFAULT_FIFO_SIZE)
      .def("start_proxy", &ProxyService::startProxy, nb::arg("blocking") = false)
      .def("stop_proxy", &ProxyService::stopProxy)
      .def("build_and_add_semaphore", &ProxyService::buildAndAddSemaphore, nb::arg("comm"), nb::arg("connection"))
      .def("add_semaphore", static_cast<SemaphoreId (ProxyService::*)(const Semaphore&)>(&ProxyService::addSemaphore),
           nb::arg("semaphore"))
      .def("add_semaphore",
           static_cast<SemaphoreId (ProxyService::*)(std::shared_ptr<Host2DeviceSemaphore>)>(
               &ProxyService::addSemaphore),
           nb::arg("semaphore"))
      .def("add_memory", &ProxyService::addMemory, nb::arg("memory"))
      .def("semaphore", &ProxyService::semaphore, nb::arg("id"))
      .def("base_port_channel", &ProxyService::basePortChannel, nb::arg("id"))
      .def("port_channel", &ProxyService::portChannel, nb::arg("id"), nb::arg("dst"), nb::arg("src"));

  nb::class_<BasePortChannel>(m, "BasePortChannel")
      .def(nb::init<>())
      .def(nb::init<SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>>(),
           nb::arg("semaphore_id"), nb::arg("semaphore"), nb::arg("proxy"))
      .def("device_handle", &BasePortChannel::deviceHandle);

  nb::class_<BasePortChannel::DeviceHandle>(m, "BasePortChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphore_id_", &BasePortChannel::DeviceHandle::semaphoreId_)
      .def_rw("semaphore_", &BasePortChannel::DeviceHandle::semaphore_)
      .def_rw("fifo_", &BasePortChannel::DeviceHandle::fifo_)
      .def_prop_ro("raw", [](const BasePortChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<PortChannel>(m, "PortChannel")
      .def(nb::init<>())
      .def(nb::init<SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>, MemoryId, MemoryId>(),
           nb::arg("semaphore_id"), nb::arg("semaphore"), nb::arg("proxy"), nb::arg("dst"), nb::arg("src"))
      .def("device_handle", &PortChannel::deviceHandle);

  nb::class_<PortChannel::DeviceHandle>(m, "PortChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphore_id_", &PortChannel::DeviceHandle::semaphoreId_)
      .def_rw("semaphore_", &PortChannel::DeviceHandle::semaphore_)
      .def_rw("fifo_", &PortChannel::DeviceHandle::fifo_)
      .def_rw("src_", &PortChannel::DeviceHandle::src_)
      .def_rw("dst_", &PortChannel::DeviceHandle::dst_)
      .def_prop_ro("raw", [](const PortChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
};
