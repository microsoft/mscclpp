#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

// #include <memory>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

class PyCudaMemory {
 public:
  PyCudaMemory(size_t size) : size_(size) { ptr_ = allocSharedPhysicalCuda<char>(size); }

  uintptr_t getPtr() const { return (uintptr_t)(ptr_.get()); }
  size_t size() const { return size_; }

 private:
  std::shared_ptr<char> ptr_;
  size_t size_;
};

void register_gpu_utils(nb::module_& m) {
  nb::class_<PyCudaMemory>(m, "PyCudaMemory")
      .def(nb::init<size_t>(), nb::arg("size"))
      .def("get_ptr", &PyCudaMemory::getPtr, "Get the raw pointer")
      .def("size", &PyCudaMemory::size, "Get the size of the allocated memory");
  m.def(
      "alloc_shared_physical_cuda", [](size_t size) { return std::make_shared<PyCudaMemory>(size); }, nb::arg("size"));
}
