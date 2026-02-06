// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// AllToAllV Python bindings for MSCCLPP
// This file provides Python bindings for the alltoallv algorithm.
// The actual implementation is in src/ext/collectives/alltoallv/

#include <Python.h>
#include <pybind11/pybind11.h>

#include <mscclpp/algorithm.hpp>

// Include the implementation header
#include "alltoallv/alltoallv_fullmesh.hpp"

namespace py = pybind11;

std::shared_ptr<mscclpp::Algorithm> createAlltoallvAlgorithm() {
  auto alltoallvAlgoBuilder = std::make_shared<mscclpp::collective::AlltoallvFullmesh>();
  return alltoallvAlgoBuilder->build();
}

void deletePtr(PyObject* capsule) {
  const char* name = PyCapsule_GetName(capsule);
  void* p = PyCapsule_GetPointer(capsule, name);
  if (p == nullptr) {
    PyErr_WriteUnraisable(capsule);
    return;
  }
  auto* ptr = static_cast<std::shared_ptr<mscclpp::Algorithm>*>(p);
  delete ptr;
}

PyObject* getCapsule(std::shared_ptr<mscclpp::Algorithm> algo) {
  auto* ptrCopy = new std::shared_ptr<mscclpp::Algorithm>(algo);
  PyObject* capsule = PyCapsule_New(ptrCopy, mscclpp::ALGORITHM_NATIVE_CAPSULE_NAME, deletePtr);
  if (capsule == nullptr) {
    delete ptrCopy;
    throw pybind11::error_already_set();
  }
  return capsule;
}

PYBIND11_MODULE(mscclpp_alltoallv, m) {
  m.doc() = "AllToAllV implementation for MSCCLPP - handles variable element counts per rank";
  m.def(
      "create_alltoallv_algorithm",
      []() { return py::reinterpret_steal<py::capsule>(getCapsule(createAlltoallvAlgorithm())); },
      "Create an alltoallv algorithm and return it as a PyCapsule usable by MSCCL++ Python bindings");
}
