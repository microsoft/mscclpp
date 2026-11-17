// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/port_channel.hpp>

#include "port_channel_handle_access.hpp"

namespace nb = nanobind;
using namespace mscclpp;

template <class Handle>
void bind_port_handle(nb::class_<Handle>& type) {
  using Access = detail::PortChannelHandleAccess;
  type.def(nb::init<>())
      .def_prop_rw(
          "semaphore_id_", [](Handle& self) { return Access::host(self).id; },
          [](Handle& self, SemaphoreId id) { Access::host(self).id = id; })
      .def_prop_rw(
          "semaphore_", [](Handle& self) -> Host2DeviceSemaphoreDeviceHandle& { return Access::hostReceive(self); },
          [](Handle& self, Host2DeviceSemaphoreDeviceHandle value) { Access::hostReceive(self) = value; },
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "fifo_", [](Handle& self) -> FifoDeviceHandle& { return Access::host(self).fifo; },
          [](Handle& self, FifoDeviceHandle value) { Access::host(self).fifo = value; },
          nb::rv_policy::reference_internal)
      .def_prop_ro("raw", [](const Handle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
}

void register_port_channel(nb::module_& m) {
  nb::class_<BaseProxyService>(m, "CppBaseProxyService")
      .def("start_proxy", &BaseProxyService::startProxy, nb::arg("blocking") = false)
      .def("stop_proxy", &BaseProxyService::stopProxy);

  nb::class_<ProxyService, BaseProxyService>(m, "CppProxyService")
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

  nb::class_<BasePortChannel>(m, "CppBasePortChannel")
      .def(nb::init<>())
      .def(nb::init<const Semaphore&>(), nb::arg("semaphore"))
      .def(nb::init<SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>>(),
           nb::arg("semaphore_id"), nb::arg("semaphore"), nb::arg("proxy"))
      .def("device_handle", &BasePortChannel::deviceHandle)
      .def("device_backend_handle", &BasePortChannel::deviceBackendHandle);

  nb::class_<BasePortChannelDeviceHandleT<PortChannelBackend::Device>>(m, "CppDeviceBasePortChannelDeviceHandle")
      .def(nb::init<>())
      .def_prop_ro("raw", [](const BasePortChannelDeviceHandleT<PortChannelBackend::Device>& self) {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<BasePortChannel::DeviceHandle> baseHandle(m, "CppBasePortChannelDeviceHandle");
  bind_port_handle(baseHandle);

  nb::class_<PortChannel>(m, "CppPortChannel")
      .def(nb::init<>())
      .def(nb::init<const Semaphore&, RegisteredMemory, RegisteredMemory>(), nb::arg("semaphore"), nb::arg("dst"),
           nb::arg("src"))
      .def(nb::init<SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>, MemoryId, MemoryId>(),
           nb::arg("semaphore_id"), nb::arg("semaphore"), nb::arg("proxy"), nb::arg("dst"), nb::arg("src"))
      .def("device_handle", &PortChannel::deviceHandle)
      .def("device_backend_handle", &PortChannel::deviceBackendHandle);

  nb::class_<PortChannelDeviceHandleT<PortChannelBackend::Device>>(m, "CppDevicePortChannelDeviceHandle")
      .def(nb::init<>())
      .def_prop_ro("raw", [](const PortChannelDeviceHandleT<PortChannelBackend::Device>& self) {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<PortChannel::DeviceHandle> portHandle(m, "CppPortChannelDeviceHandle");
  bind_port_handle(portHandle);
  using Access = detail::PortChannelHandleAccess;
  portHandle
      .def_prop_rw(
          "src_", [](const PortChannelDeviceHandle& self) { return Access::hostBound(self).src; },
          [](PortChannelDeviceHandle& self, MemoryId id) { Access::hostBound(self).src = id; })
      .def_prop_rw(
          "dst_", [](const PortChannelDeviceHandle& self) { return Access::hostBound(self).dst; },
          [](PortChannelDeviceHandle& self, MemoryId id) { Access::hostBound(self).dst = id; });
};
