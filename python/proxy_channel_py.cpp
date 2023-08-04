// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/proxy_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_proxy_channel(nb::module_& m) {
  nb::class_<BaseProxyService>(m, "BaseProxyService")
      .def("start_proxy", &BaseProxyService::startProxy)
      .def("stop_proxy", &BaseProxyService::stopProxy);

  nb::class_<ProxyService, BaseProxyService>(m, "ProxyService")
      .def(nb::init<Communicator&>(), nb::arg("comm"))
      .def("start_proxy", &ProxyService::startProxy)
      .def("stop_proxy", &ProxyService::stopProxy)
      .def("build_and_add_semaphore", &ProxyService::buildAndAddSemaphore, nb::arg("connection"))
      .def("add_semaphore", &ProxyService::addSemaphore, nb::arg("semaphore"))
      .def("add_memory", &ProxyService::addMemory, nb::arg("memory"))
      .def("semaphore", &ProxyService::semaphore, nb::arg("id"))
      .def("proxy_channel", &ProxyService::proxyChannel, nb::arg("id"));

  nb::class_<ProxyChannel>(m, "ProxyChannel")
      .def(nb::init<SemaphoreId, Host2DeviceSemaphore::DeviceHandle, FifoDeviceHandle>(), nb::arg("semaphoreId"),
           nb::arg("semaphore"), nb::arg("fifo"));

  nb::class_<SimpleProxyChannel>(m, "SimpleProxyChannel")
      .def(nb::init<ProxyChannel, MemoryId, MemoryId>(), nb::arg("proxyChan"), nb::arg("dst"), nb::arg("src"))
      .def(nb::init<SimpleProxyChannel>(), nb::arg("proxyChan"));

  m.def("device_handle", &deviceHandle<ProxyChannel>, nb::arg("proxyChannel"));
  m.def("device_handle", &deviceHandle<SimpleProxyChannel>, nb::arg("simpleProxyChannel"));
};
