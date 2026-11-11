// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
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
  } else if (type == "torch.float64") {
    return DLDataType{kDLFloat, 64, 1};
  } else if (type == "torch.int8") {
    return DLDataType{kDLInt, 8, 1};
  } else if (type == "torch.int16") {
    return DLDataType{kDLInt, 16, 1};
  } else if (type == "torch.int32") {
    return DLDataType{kDLInt, 32, 1};
  } else if (type == "torch.int64") {
    return DLDataType{kDLInt, 64, 1};
  } else if (type == "torch.uint16") {
    return DLDataType{kDLUInt, 16, 1};
  } else if (type == "torch.uint32") {
    return DLDataType{kDLUInt, 32, 1};
  } else if (type == "torch.uint64") {
    return DLDataType{kDLUInt, 64, 1};
  } else if (type == "torch.bool") {
    return DLDataType{kDLBool, 8, 1};
  } else if (type == "torch.complex64") {
    return DLDataType{kDLComplex, 64, 1};
  } else if (type == "torch.complex128") {
    return DLDataType{kDLComplex, 128, 1};
  } else if (type == "torch.bfloat16") {
    return DLDataType{kDLBfloat, 16, 1};
  } else if (type == "torch.float16") {
    return DLDataType{kDLFloat, 16, 1};
  } else if (type == "torch.float8_e4m3fn") {
    return DLDataType{kDLFloat8_e4m3fn, 8, 1};
  } else if (type == "torch.float8_e4m3fnuz") {
    return DLDataType{kDLFloat8_e4m3fnuz, 8, 1};
  } else if (type == "torch.float8_e5m2") {
    return DLDataType{kDLFloat8_e5m2, 8, 1};
  } else if (type == "torch.float8_e5m2fnuz") {
    return DLDataType{kDLFloat8_e5m2fnuz, 8, 1};
  } else if (type == "torch.uint8") {
    return DLDataType{kDLUInt, 8, 1};
  } else if (type == "fp8_e4m3b15") {
    // No standard DLPack code for fp8_e4m3b15; store as raw uint8 bytes.
    return DLDataType{kDLUInt, 8, 1};
  } else {
    throw Error("Unsupported type: " + type, ErrorCode::InvalidUsage);
  }
}

struct DlpackTensor {
  DLManagedTensor tensor{};
  std::shared_ptr<void> owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

static nb::capsule toDlpack(void* data, int deviceId, std::shared_ptr<void> owner, DLDataType dtype,
                            std::vector<int64_t> shape, std::vector<int64_t> strides) {
  if (!strides.empty() && strides.size() != shape.size()) {
    throw nb::value_error("Shape and strides must have the same number of dimensions");
  }
  auto managed = std::make_unique<DlpackTensor>();
  managed->owner = std::move(owner);
  managed->shape = std::move(shape);
  managed->strides = std::move(strides);
  DLManagedTensor* dlManagedTensor = &managed->tensor;
  dlManagedTensor->dl_tensor.data = data;
  dlManagedTensor->dl_tensor.device.device_type = getDeviceType();
  dlManagedTensor->dl_tensor.device.device_id = deviceId;
  dlManagedTensor->dl_tensor.ndim = managed->shape.size();
  dlManagedTensor->dl_tensor.strides = managed->strides.empty() ? nullptr : managed->strides.data();
  dlManagedTensor->dl_tensor.shape = managed->shape.data();
  dlManagedTensor->dl_tensor.byte_offset = 0;
  dlManagedTensor->dl_tensor.dtype = dtype;
  dlManagedTensor->manager_ctx = managed.get();
  dlManagedTensor->deleter = [](DLManagedTensor* self) { delete static_cast<DlpackTensor*>(self->manager_ctx); };

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
  if (dlCapsule == nullptr) {
    throw nb::python_error();
  }
  managed.release();
  return nb::steal<nb::capsule>(dlCapsule);
}

void register_gpu_utils(nb::module_& m) {
  m.attr("is_hip") = getDeviceType() == kDLROCM;
  m.def("is_nvls_supported", &isNvlsSupported);
  m.def("is_bulk_supported", &isBulkSupported);

  m.def(
      "cpp_gpu_buffer_to_dlpack",
      [](uintptr_t ptr, std::string dataType, std::vector<int64_t> shape, std::vector<int64_t> strides, int deviceId,
         nb::object owner) {
        auto keepAlive = std::shared_ptr<nb::object>(new nb::object(std::move(owner)), [](nb::object* value) {
          nb::gil_scoped_acquire gil;
          delete value;
        });
        return toDlpack(reinterpret_cast<void*>(ptr), deviceId, std::move(keepAlive), getDlType(dataType),
                        std::move(shape), std::move(strides));
      },
      nb::arg("ptr"), nb::arg("data_type"), nb::arg("shape"), nb::arg("strides"), nb::arg("device_id"),
      nb::arg("owner"));

  nb::enum_<GpuBufferGranularity>(m, "CppGpuBufferGranularity")
      .value("MultiCastMinimum", GpuBufferGranularity::MultiCastMinimum)
      .value("MultiCastRecommended", GpuBufferGranularity::MultiCastRecommended);

  nb::class_<GpuBuffer<char>>(m, "CppRawGpuBuffer")
      .def(nb::init<size_t, GpuBufferGranularity>(), nb::arg("nelems"),
           nb::arg("granularity") = GpuBufferGranularity::MultiCastMinimum)
      .def("nelems", &GpuBuffer<char>::nelems)
      .def("bytes", &GpuBuffer<char>::bytes)
      .def("data", [](GpuBuffer<char>& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("device_id", &GpuBuffer<char>::deviceId)
      .def(
          "to_dlpack",
          [](GpuBuffer<char>& self, std::string dataType, std::vector<int64_t> shape, std::vector<int64_t> strides) {
            DLDataType dtype = getDlType(dataType);
            if (shape.empty()) {
              shape.push_back(self.nelems() / ((dtype.bits * dtype.lanes + 7) / BYTE_BITS));
            }
            return toDlpack(self.data(), self.deviceId(), self.memory(), dtype, std::move(shape), std::move(strides));
          },
          nb::arg("data_type"), nb::arg("shape") = std::vector<int64_t>(), nb::arg("strides") = std::vector<int64_t>());
}
