// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace nb = nanobind;
using namespace mscclpp;

constexpr int BYTE_BITS = 8;

struct DlpackContext {
  DLManagedTensor managedTensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::shared_ptr<void> owner;
};

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

static void dlpackDeleter(DLManagedTensor* self) { delete static_cast<DlpackContext*>(self->manager_ctx); }

static void dlpackCapsuleDestructor(PyObject* capsule) {
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
}

static nb::capsule makeDlpack(void* data, size_t bytes, int deviceId, std::shared_ptr<void> owner, std::string dataType,
                              std::vector<int64_t> shape, std::vector<int64_t> strides) {
  DLDataType dtype = getDlType(dataType);
  auto ctx = std::make_unique<DlpackContext>();
  if (shape.empty()) {
    ctx->shape.push_back((int64_t)(bytes / ((dtype.bits * dtype.lanes + 7) / BYTE_BITS)));
  } else {
    ctx->shape = std::move(shape);
  }
  ctx->strides = std::move(strides);
  ctx->owner = std::move(owner);

  DLManagedTensor* dlManagedTensor = &ctx->managedTensor;
  dlManagedTensor->dl_tensor.data = data;
  dlManagedTensor->dl_tensor.device.device_type = getDeviceType();
  dlManagedTensor->dl_tensor.device.device_id = deviceId;
  dlManagedTensor->dl_tensor.ndim = ctx->shape.size();
  dlManagedTensor->dl_tensor.strides = ctx->strides.empty() ? nullptr : ctx->strides.data();
  dlManagedTensor->dl_tensor.shape = ctx->shape.data();
  dlManagedTensor->dl_tensor.byte_offset = 0;
  dlManagedTensor->dl_tensor.dtype = dtype;
  dlManagedTensor->manager_ctx = ctx.get();
  dlManagedTensor->deleter = dlpackDeleter;

  PyObject* dlCapsule = PyCapsule_New(static_cast<void*>(dlManagedTensor), "dltensor", dlpackCapsuleDestructor);
  if (dlCapsule == nullptr) {
    throw Error("Failed to create DLPack capsule.", ErrorCode::InvalidUsage);
  }
  ctx.release();
  return nb::steal<nb::capsule>(dlCapsule);
}

static nb::capsule toDlpack(GpuBuffer<char> buffer, std::string dataType, std::vector<int64_t>& shape,
                            std::vector<int64_t>& strides) {
  auto owner = std::make_shared<GpuBuffer<char>>(buffer);
  return makeDlpack(buffer.data(), buffer.nelems(), buffer.deviceId(), std::move(owner), dataType, shape, strides);
}

static nb::capsule toDlpack(std::shared_ptr<GpuBufferPoolAllocation> allocation, std::string dataType,
                            std::vector<int64_t>& shape, std::vector<int64_t>& strides) {
  void* data = allocation->data();
  size_t bytes = allocation->bytes();
  int deviceId = allocation->deviceId();
  return makeDlpack(data, bytes, deviceId, std::move(allocation), dataType, shape, strides);
}

void register_gpu_utils(nb::module_& m) {
  m.def("is_nvls_supported", &isNvlsSupported);

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
            return toDlpack(self, dataType, shape, strides);
          },
          nb::arg("data_type"), nb::arg("shape") = std::vector<int64_t>(), nb::arg("strides") = std::vector<int64_t>());

  nb::class_<GpuBufferPoolAllocation>(m, "CppRawGpuBufferPoolAllocation")
      .def("bytes", &GpuBufferPoolAllocation::bytes)
      .def("offset", &GpuBufferPoolAllocation::offset)
      .def("data", [](GpuBufferPoolAllocation& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("device_id", &GpuBufferPoolAllocation::deviceId)
      .def(
          "to_dlpack",
          [](GpuBufferPoolAllocation& self, std::string dataType, std::vector<int64_t> shape,
             std::vector<int64_t> strides) { return toDlpack(self.shared_from_this(), dataType, shape, strides); },
          nb::arg("data_type"), nb::arg("shape") = std::vector<int64_t>(), nb::arg("strides") = std::vector<int64_t>());

  nb::class_<GpuBufferPool>(m, "CppRawGpuBufferPool")
      .def(nb::init<size_t, GpuBufferGranularity>(), nb::arg("bytes"),
           nb::arg("granularity") = GpuBufferGranularity::MultiCastMinimum)
      .def("bytes", &GpuBufferPool::bytes)
      .def("free_bytes", &GpuBufferPool::freeBytes)
      .def("active_bytes", &GpuBufferPool::activeBytes)
      .def("data", [](GpuBufferPool& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def("device_id", &GpuBufferPool::deviceId)
      .def("allocate", &GpuBufferPool::allocate, nb::arg("bytes"), nb::arg("alignment") = 256,
           nb::arg("alloc_id") = std::nullopt);
}
