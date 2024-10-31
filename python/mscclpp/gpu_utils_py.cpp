#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

// #include <memory>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

class PyCudaMemory {
 public:
  PyCudaMemory(size_t size, DataType dtype) : size_(size), dtype_(dtype) {
    ptr_ = allocSharedPhysicalCudaPtr<char>(size);
  }

  uintptr_t getPtr() const { return (uintptr_t)(ptr_.get()); }
  size_t size() const { return size_; }
  DataType dtype() const { return dtype_; }

 private:
  std::shared_ptr<char> ptr_;
  size_t size_;
  DataType dtype_;
};

std::shared_ptr<PyCudaMemory> allocSharedPhysicalCudaPtrDispatcher(size_t count, DataType dtype) {
  size_t size = 0;
  switch (dtype) {
    case DataType::FLOAT32:
      size = count * sizeof(float);
      break;
    case DataType::FLOAT16:
      size = count * sizeof(__half);
      break;
    case DataType::BFLOAT16:
      size = count * sizeof(__bfloat16);
      break;
    case DataType::INT32:
      size = count * sizeof(int);
      break;
    default:
      throw std::runtime_error("Unsupported data type.");
  }

  return std::make_shared<PyCudaMemory>(size, dtype);
}

void register_gpu_utils(nb::module_& m) {
  nb::class_<PyCudaMemory>(m, "PyCudaMemory")
      .def(nb::init<size_t, DataType>(), nb::arg("size"), nb::arg("dtype"))
      .def("get_ptr", &PyCudaMemory::getPtr, "Get the raw pointer")
      .def("size", &PyCudaMemory::size, "Get the size of the allocated memory")
      .def("dtype", &PyCudaMemory::dtype, "Get the data type of the memory");
  m.def("alloc_shared_physical_cuda_ptr", &allocSharedPhysicalCudaPtrDispatcher, nb::arg("count"), nb::arg("dtype"));
}
