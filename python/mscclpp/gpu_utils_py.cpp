#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

static DLDeviceType getDeviceType() {
#if defined(__HIP_PLATFORM_AMD__)
  return kDLROCm;
#else
  return kDLCUDA;
#endif
}

static DLDataType getDlType(std::string type) {
  if (type == "float") {
    return DLDataType{kDLFloat, 32, 1};
  } else if (type == "int") {
    return DLDataType{kDLInt, 32, 1};
  } else if (type == "uint") {
    return DLDataType{kDLUInt, 32, 1};
  } else if (type == "bfloat") {
    return DLDataType{kDLBfloat, 16, 1};
  } else if (type == "float16") {
    return DLDataType{kDLFloat, 16, 1};
  } else {
    throw Error("Unsupported type: " + type, ErrorCode::InvalidUsage);
  }
}

static nb::capsule toDlpack(GpuBuffer<char> buffer, std::string type) {
  int64_t* shape = new int64_t[1];
  DLDataType dtype = getDlType(type);
  shape[0] = buffer.nelems() / (dtype.bits / sizeof(char));

  DLManagedTensor* dlManagedTensor = new DLManagedTensor();
  dlManagedTensor->dl_tensor.data = buffer.data();
  dlManagedTensor->dl_tensor.device.device_type = getDeviceType();
  dlManagedTensor->dl_tensor.device.device_id = buffer.deviceId();
  dlManagedTensor->dl_tensor.ndim = 1;
  dlManagedTensor->dl_tensor.strides = nullptr;
  dlManagedTensor->dl_tensor.shape = shape;
  dlManagedTensor->dl_tensor.byte_offset = 0;
  dlManagedTensor->dl_tensor.dtype = getDlType(type);
  dlManagedTensor->manager_ctx = new GpuBuffer<char>(buffer);
  dlManagedTensor->deleter = [](DLManagedTensor* self) {
    auto buffer = static_cast<GpuBuffer<char>*>(self->manager_ctx);
    delete buffer;
    self->dl_tensor.data = nullptr;
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      self->dl_tensor.data = nullptr;
    }
    delete self;
  };

  return nb::capsule(dlManagedTensor, [](void* self) noexcept {
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(self);
    auto buffer = static_cast<GpuBuffer<char>*>(tensor->manager_ctx);
    delete buffer;
    tensor->dl_tensor.data = nullptr;
    if (tensor->dl_tensor.shape != nullptr) {
      delete[] tensor->dl_tensor.shape;
      tensor->dl_tensor.data = nullptr;
    }
    delete tensor;
  });
}

void register_gpu_utils(nb::module_& m) {
  m.def("is_nvls_supported", &isNvlsSupported);

  nb::class_<GpuBuffer<char>>(m, "RawGpuBuffer")
      .def(nb::init<size_t>(), nb::arg("nelems"))
      .def("nelems", &GpuBuffer<char>::nelems)
      .def("bytes", &GpuBuffer<char>::bytes)
      .def("data", [](GpuBuffer<char>& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("device_id", &GpuBuffer<char>::deviceId)
      .def("to_dlpack", &toDlpack, nb::arg("type"));
}
