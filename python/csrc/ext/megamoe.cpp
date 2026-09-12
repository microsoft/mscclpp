// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nanobind/nanobind.h>

#ifdef MSCCLPP_BUILD_EXT_MEGAMOE
#include <dlpack/dlpack.h>
#include <nanobind/stl/shared_ptr.h>

#include <array>
#include <memory>
#include <stdexcept>

#include "megamoe.hpp"
#endif

namespace nb = nanobind;

#ifdef MSCCLPP_BUILD_EXT_MEGAMOE
namespace {
using namespace mscclpp::megamoe;

struct WorkspaceTensor {
  DLManagedTensor tensor{};
  std::shared_ptr<MegaMoeContext> owner;
  std::array<int64_t, 2> shape;
};

nb::capsule inputDlpack(std::shared_ptr<MegaMoeContext> context, int tokens) {
  if (tokens < 0 || tokens > context->config().maxTokens)
    throw std::invalid_argument("input_view token count exceeds MegaMoE capacity");
  auto view = std::make_unique<WorkspaceTensor>();
  view->shape = {tokens, context->config().hidden};
  view->owner = std::move(context);
  view->tensor.dl_tensor.data = view->owner->input();
  view->tensor.dl_tensor.device = {kDLCUDA, view->owner->device()};
  view->tensor.dl_tensor.ndim = 2;
  view->tensor.dl_tensor.dtype = {kDLBfloat, 16, 1};
  view->tensor.dl_tensor.shape = view->shape.data();
  view->tensor.manager_ctx = view.get();
  view->tensor.deleter = [](DLManagedTensor* tensor) { delete static_cast<WorkspaceTensor*>(tensor->manager_ctx); };
  PyObject* capsule = PyCapsule_New(&view->tensor, "dltensor", [](PyObject* object) noexcept {
    if (!PyCapsule_IsValid(object, "dltensor")) return;
    auto tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(object, "dltensor"));
    if (tensor && tensor->deleter) tensor->deleter(tensor);
  });
  if (!capsule) throw nb::python_error();
  view.release();
  return nb::steal<nb::capsule>(capsule);
}
}  // namespace
#endif

void register_megamoe(nb::module_& m) {
#ifdef MSCCLPP_BUILD_EXT_MEGAMOE
  using namespace mscclpp::megamoe;
  m.def("megamoe_available", []() { return true; });
  nb::class_<NativeConfig>(m, "CppMegaMoeConfig")
      .def(nb::init<>())
      .def_rw("rank", &NativeConfig::rank)
      .def_rw("world_size", &NativeConfig::worldSize)
      .def_rw("max_tokens", &NativeConfig::maxTokens)
      .def_rw("hidden", &NativeConfig::hidden)
      .def_rw("intermediate", &NativeConfig::intermediate)
      .def_rw("num_experts", &NativeConfig::numExperts)
      .def_rw("top_k", &NativeConfig::topK)
      .def_rw("sm_margin", &NativeConfig::smMargin)
      .def_rw("weight_e5m2", &NativeConfig::weightE5M2)
      .def_rw("gate_up_clamp", &NativeConfig::gateUpClamp);
  nb::class_<MegaMoeContext>(m, "CppMegaMoeContext")
      .def_static(
          "create",
          [](std::shared_ptr<mscclpp::Communicator> comm, const NativeConfig& config, uintptr_t fc1, uintptr_t fc1Scale,
             uintptr_t fc2, uintptr_t fc2Scale, uintptr_t stream, int tag) {
            PackedWeights weights{reinterpret_cast<uint8_t*>(fc1), reinterpret_cast<uint8_t*>(fc1Scale),
                                  reinterpret_cast<uint8_t*>(fc2), reinterpret_cast<uint8_t*>(fc2Scale)};
            return std::make_shared<MegaMoeContext>(std::move(comm), config, weights,
                                                    reinterpret_cast<cudaStream_t>(stream), tag);
          },
          nb::arg("communicator"), nb::arg("config"), nb::arg("fc1"), nb::arg("fc1_scale"), nb::arg("fc2"),
          nb::arg("fc2_scale"), nb::arg("stream") = 0, nb::arg("tag") = 17920, nb::call_guard<nb::gil_scoped_release>())
      .def("input_ptr", [](const MegaMoeContext& self) { return reinterpret_cast<uintptr_t>(self.input()); })
      .def("input_dlpack", &inputDlpack, nb::arg("num_tokens"))
      .def_prop_ro("device", &MegaMoeContext::device)
      .def_prop_ro("cta_count", &MegaMoeContext::ctaCount)
      .def_prop_ro("shared_bytes", &MegaMoeContext::sharedBytes)
      .def_prop_ro("symmetric_bytes", &MegaMoeContext::symmetricBytes)
      .def_prop_ro("private_bytes", &MegaMoeContext::privateBytes)
      .def(
          "forward",
          [](MegaMoeContext& self, uintptr_t input, uintptr_t ids, uintptr_t scores, uintptr_t output, int tokens,
             uintptr_t stream) {
            self.forward(reinterpret_cast<const void*>(input), reinterpret_cast<const int32_t*>(ids),
                         reinterpret_cast<const float*>(scores), reinterpret_cast<void*>(output), tokens,
                         reinterpret_cast<cudaStream_t>(stream));
          },
          nb::arg("input"), nb::arg("topk_ids"), nb::arg("topk_weights"), nb::arg("output"), nb::arg("num_tokens"),
          nb::arg("stream"));
#else
  m.def("megamoe_available", []() { return false; });
#endif
}
