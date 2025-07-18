// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/memory_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_memory_channel(nb::module_& m) {
  nb::class_<BaseMemoryChannel>(m, "BaseMemoryChannel")
      .def(nb::init<>())
      .def(nb::init<std::shared_ptr<MemoryDevice2DeviceSemaphore>>(), nb::arg("semaphore"))
      .def(nb::init<const Semaphore&>(), nb::arg("semaphore"))
      .def("device_handle", &BaseMemoryChannel::deviceHandle);

  nb::class_<BaseMemoryChannel::DeviceHandle>(m, "BaseMemoryChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphore_", &BaseMemoryChannel::DeviceHandle::semaphore_)
      .def_prop_ro("raw", [](const BaseMemoryChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });

  nb::class_<MemoryChannel>(m, "MemoryChannel")
      .def(nb::init<>())
      .def("__init__",
           [](MemoryChannel* memoryChannel, std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
              RegisteredMemory dst,
              uintptr_t src) { new (memoryChannel) MemoryChannel(semaphore, dst, reinterpret_cast<void*>(src)); })
      .def("__init__",
           [](MemoryChannel* memoryChannel, std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
              RegisteredMemory dst, uintptr_t src, uintptr_t packet_buffer) {
             new (memoryChannel)
                 MemoryChannel(semaphore, dst, reinterpret_cast<void*>(src), reinterpret_cast<void*>(packet_buffer));
           })
      .def("device_handle", &MemoryChannel::deviceHandle);

  nb::class_<MemoryChannel::DeviceHandle>(m, "MemoryChannelDeviceHandle")
      .def(nb::init<>())
      .def_rw("semaphore_", &MemoryChannel::DeviceHandle::semaphore_)
      .def_rw("dst_", &MemoryChannel::DeviceHandle::dst_)
      .def_rw("src_", &MemoryChannel::DeviceHandle::src_)
      .def_rw("packetBuffer_", &MemoryChannel::DeviceHandle::packetBuffer_)
      .def_prop_ro("raw", [](const MemoryChannel::DeviceHandle& self) -> nb::bytes {
        return nb::bytes(reinterpret_cast<const char*>(&self), sizeof(self));
      });
};
