#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

class PyGpuBuffer {
 public:
  PyGpuBuffer(size_t size) : size_(size) { ptr_ = gpuMemAlloc<char>(size); }

  uintptr_t ptr() const { return (uintptr_t)(ptr_.get()); }
  size_t size() const { return size_; }

 private:
  std::shared_ptr<char> ptr_;
  size_t size_;
};

void register_gpu_utils(nb::module_& m) {
  nb::class_<PyGpuBuffer>(m, "PyGpuBuffer")
      .def(nb::init<size_t>(), nb::arg("size"))
      .def("ptr", &PyGpuBuffer::ptr, "Get the address of the allocated memory")
      .def("size", &PyGpuBuffer::size, "Get the size of the allocated memory");
}
