// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>

namespace nb = nanobind;
using namespace mscclpp;
using namespace mscclpp::collective;

using AlgoMap = std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>>;

static AlgoSelectFunc wrapPySelector(nb::callable pyFunc) {
  auto shared = std::make_shared<nb::callable>(std::move(pyFunc));
  return [shared](const AlgoMap& algoMap,
                  const CollectiveRequest& request) -> std::shared_ptr<Algorithm> {
    nb::gil_scoped_acquire gil;
    nb::object result = (*shared)(nb::cast(algoMap, nb::rv_policy::reference),
                                  nb::cast(request, nb::rv_policy::reference));
    if (result.is_none()) return nullptr;
    return nb::cast<std::shared_ptr<Algorithm>>(result);
  };
}

void register_algorithm_collection_builder(nb::module_& m) {
  nb::class_<AlgorithmCollectionBuilder>(m, "CppAlgorithmCollectionBuilder")
      .def_static("get_instance", &AlgorithmCollectionBuilder::getInstance)
      .def("add_algorithm_builder", &AlgorithmCollectionBuilder::addAlgorithmBuilder, nb::arg("builder"))
      .def(
          "add_dsl_algorithm_builder",
          [](AlgorithmCollectionBuilder& self, std::shared_ptr<DslAlgorithm> algorithm) {
            self.addAlgorithmBuilder(algorithm);
          },
          nb::arg("algorithm"))
      .def(
          "set_algorithm_selector",
          [](AlgorithmCollectionBuilder& self, nb::callable selector) {
            self.setAlgorithmSelector(wrapPySelector(std::move(selector)));
          },
          nb::arg("selector"))
      .def(
          "set_fallback_algorithm_selector",
          [](AlgorithmCollectionBuilder& self, nb::callable selector) {
            self.setFallbackAlgorithmSelector(wrapPySelector(std::move(selector)));
          },
          nb::arg("selector"))
      .def("build", &AlgorithmCollectionBuilder::build)
      .def("build_default_algorithms", &AlgorithmCollectionBuilder::buildDefaultAlgorithms, nb::arg("scratch_buffer"),
           nb::arg("scratch_buffer_size"), nb::arg("flag_buffer"), nb::arg("flag_buffer_size"), nb::arg("rank"))
      .def_static("reset", &AlgorithmCollectionBuilder::reset);
}