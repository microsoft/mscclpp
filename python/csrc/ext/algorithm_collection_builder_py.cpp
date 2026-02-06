// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <mscclpp/algorithm.hpp>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>

namespace nb = nanobind;
using namespace mscclpp;
using namespace mscclpp::collective;

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
      .def("set_algorithm_selector", &AlgorithmCollectionBuilder::setAlgorithmSelector, nb::arg("selector"))
      .def("set_fallback_algorithm_selector", &AlgorithmCollectionBuilder::setFallbackAlgorithmSelector,
           nb::arg("selector"))
      .def("build", &AlgorithmCollectionBuilder::build)
      .def("build_default_algorithms", &AlgorithmCollectionBuilder::buildDefaultAlgorithms, nb::arg("scratch_buffer"),
           nb::arg("scratch_buffer_size"), nb::arg("rank"))
      .def_static("reset", &AlgorithmCollectionBuilder::reset);
}