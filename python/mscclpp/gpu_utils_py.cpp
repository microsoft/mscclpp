#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

constexpr int BYTE_BITS = 8;

static DLDeviceType getDeviceType() {
#if defined(__HIP_PLATFORM_AMD__)
  return kDLROCm;
#else
  return kDLCUDA;
#endif
}

static DLDataType getDlType(std::string type) {
  if (type == "torch.float") {
    return DLDataType{kDLFloat, 32, 1};
  } else if (type == "torch.int") {
    return DLDataType{kDLInt, 32, 1};
  } else if (type == "torch.uint32") {
    return DLDataType{kDLUInt, 32, 1};
  } else if (type == "torch.bfloat16") {
    return DLDataType{kDLBfloat, 16, 1};
  } else if (type == "torch.float16") {
    return DLDataType{kDLFloat, 16, 1};
  } else {
    throw Error("Unsupported type: " + type, ErrorCode::InvalidUsage);
  }
}

static nb::capsule toDlpack(GpuBuffer<char> buffer, std::string dataType) {
  int64_t* shape = new int64_t[1];
  DLDataType dtype = getDlType(dataType);
  shape[0] = buffer.nelems() / ((dtype.bits * dtype.lanes + 7) / BYTE_BITS);

  DLManagedTensor* dlManagedTensor = new DLManagedTensor();
  dlManagedTensor->dl_tensor.data = buffer.data();
  dlManagedTensor->dl_tensor.device.device_type = getDeviceType();
  dlManagedTensor->dl_tensor.device.device_id = buffer.deviceId();
  dlManagedTensor->dl_tensor.ndim = 1;
  dlManagedTensor->dl_tensor.strides = nullptr;
  dlManagedTensor->dl_tensor.shape = shape;
  dlManagedTensor->dl_tensor.byte_offset = 0;
  dlManagedTensor->dl_tensor.dtype = dtype;
  dlManagedTensor->manager_ctx = new GpuBuffer<char>(buffer);
  dlManagedTensor->deleter = [](DLManagedTensor* self) {
    delete static_cast<GpuBuffer<char>*>(self->manager_ctx);
    self->manager_ctx = nullptr;
    self->dl_tensor.data = nullptr;
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      self->dl_tensor.shape = nullptr;
    }
    delete self;
  };

  return nb::capsule(dlManagedTensor, "dltensor", [](void* self) noexcept {
    nb::capsule* capsule = static_cast<nb::capsule*>(self);
    if (strcmp(capsule->name(), "dltensor") != 0) {
      return;
    }
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(capsule->data());
    if (tensor == nullptr) {
      return;
    }
    if (tensor->deleter) {
      tensor->deleter(tensor);
    }
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
      .def("to_dlpack", [](GpuBuffer<char>& self, std::string dataType) { return toDlpack(self, dataType); });
}
