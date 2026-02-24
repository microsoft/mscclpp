// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

constexpr int BYTE_BITS = 8;

static DLDeviceType getDeviceType() {
#if defined(MSCCLPP_USE_ROCM)
  return kDLROCM;
#else
  return kDLCUDA;
#endif
}

static DLDataType getDlType(std::string type) {
  if (type == "torch.float32") {
    return DLDataType{kDLFloat, 32, 1};
  } else if (type == "torch.int32") {
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

static nb::capsule toDlpack(GpuBuffer<char> buffer, std::string dataType, std::vector<int64_t>& shape,
                            std::vector<int64_t>& strides) {
  DLDataType dtype = getDlType(dataType);
  int64_t* tensorShape = shape.size() > 0 ? new int64_t[shape.size()] : new int64_t[1];
  int64_t* tensorStrides = strides.size() > 0 ? new int64_t[strides.size()] : nullptr;
  if (shape.size() == 0) {
    tensorShape[0] = (int64_t)(buffer.nelems() / ((dtype.bits * dtype.lanes + 7) / BYTE_BITS));
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      tensorShape[i] = shape[i];
    }
  }
  for (size_t i = 0; i < strides.size(); ++i) {
    tensorStrides[i] = strides[i];
  }

  DLManagedTensor* dlManagedTensor = new DLManagedTensor();
  dlManagedTensor->dl_tensor.data = buffer.data();
  dlManagedTensor->dl_tensor.device.device_type = getDeviceType();
  dlManagedTensor->dl_tensor.device.device_id = buffer.deviceId();
  dlManagedTensor->dl_tensor.ndim = shape.size() == 0 ? 1 : shape.size();
  dlManagedTensor->dl_tensor.strides = tensorStrides;
  dlManagedTensor->dl_tensor.shape = tensorShape;
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
      if (self->dl_tensor.strides) {
        delete[] self->dl_tensor.strides;
        self->dl_tensor.strides = nullptr;
      }
    }
    delete self;
  };

  PyObject* dlCapsule = PyCapsule_New(static_cast<void*>(dlManagedTensor), "dltensor", [](PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "used_dltensor")) {
      return;
    }
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
      return;
    }
    DLManagedTensor* managedTensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managedTensor == nullptr) {
      return;
    }
    if (managedTensor->deleter) {
      managedTensor->deleter(managedTensor);
    }
  });
  return nb::steal<nb::capsule>(dlCapsule);
}

bool builtWithIb() {
#if defined(USE_IBVERBS)
  return true;
#else
  return false;
#endif
}

void register_gpu_utils(nb::module_& m) {
  m.def("is_nvls_supported", &isNvlsSupported);
  m.def("built_with_ib", &builtWithIb);

  nb::class_<GpuBuffer<char>>(m, "CppRawGpuBuffer")
      .def(nb::init<size_t>(), nb::arg("nelems"))
      .def("nelems", &GpuBuffer<char>::nelems)
      .def("bytes", &GpuBuffer<char>::bytes)
      .def("data", [](GpuBuffer<char>& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("device_id", &GpuBuffer<char>::deviceId)
      .def(
          "to_dlpack",
          [](GpuBuffer<char>& self, std::string dataType, std::vector<int64_t> shape, std::vector<int64_t> strides) {
            return toDlpack(self, dataType, shape, strides);
          },
          nb::arg("data_type"), nb::arg("shape") = std::vector<int64_t>(), nb::arg("strides") = std::vector<int64_t>());
}
