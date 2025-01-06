#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

void register_gpu_utils(nb::module_& m) {
  m.def("is_nvls_supported", &isNvlsSupported);

  nb::class_<GpuBuffer<char>>(m, "RawGpuBuffer")
      .def(nb::init<size_t>(), nb::arg("nelems"))
      .def("nelems", &GpuBuffer<char>::nelems)
      .def("bytes", &GpuBuffer<char>::bytes)
      .def("data", [](GpuBuffer<char>& self) { return reinterpret_cast<uintptr_t>(self.data()); });
}
