// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/memory_channel.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_memory_channel(nb::module_& m) {
  nb::class_<MemoryChannel> memoryChannel(m, "MemoryChannel");
  memoryChannel
      .def("__init__",
           [](MemoryChannel* memoryChannel, std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
              RegisteredMemory dst, uintptr_t src) { new (memoryChannel) MemoryChannel(semaphore, dst, (void*)src); })
      .def("__init__",
           [](MemoryChannel* memoryChannel, std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
              RegisteredMemory dst, uintptr_t src, uintptr_t get_packet_buffer) {
             new (memoryChannel) MemoryChannel(semaphore, dst, (void*)src, (void*)get_packet_buffer);
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
