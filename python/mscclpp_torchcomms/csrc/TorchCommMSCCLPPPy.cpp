// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Pybind11 module exposing TorchCommMSCCLPP to Python.
//
// This is intentionally minimal — TorchCommMSCCLPP is used via the TorchCommBackend
// C++ interface (polymorphic dispatch). The Python binding just makes the class
// constructible so torchcomms' Python layer can instantiate it. All collective
// methods are called through the base class interface, not through Python bindings
// of individual methods.
//
// The extern "C" create_dynamic_loader_mscclpp() function is the dynamic loader
// interface required by torchcomms v0.2.0's TorchCommFactory. When torchcomms
// dlopen's our .so, it looks up this symbol to get function pointers for
// creating/destroying backend instances and checking ABI version compatibility.
//
// User-defined algorithms: configure via mscclpp.AlgorithmCollectionBuilder
// BEFORE creating the TorchComms communicator. The backend picks up whatever
// is registered on the builder singleton during init().

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "TorchCommMSCCLPP.hpp"

namespace py = pybind11;
using namespace torch::comms;

// --- Dynamic loader interface for torchcomms TorchCommFactory ---
//
// TorchComms discovers backends by dlopen'ing the .so pointed to by
// TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP, then calling:
//   dlsym(handle, "create_dynamic_loader_mscclpp")
// The function name encodes the backend name ("mscclpp"). The returned
// DynamicLoaderInterface provides function pointers for creating/destroying
// backend instances and checking ABI version compatibility.

static TorchCommBackend* new_comm_impl() { return new TorchCommMSCCLPP(); }

static void destroy_comm_impl(TorchCommBackend* comm) { delete comm; }

static const char* get_supported_version_impl() { return TORCHCOMM_BACKEND_ABI_VERSION; }

extern "C" __attribute__((visibility("default"))) DynamicLoaderInterface create_dynamic_loader_mscclpp() {
  return DynamicLoaderInterface{
      .new_comm = new_comm_impl,
      .destroy_comm = destroy_comm_impl,
      .get_supported_version = get_supported_version_impl,
  };
}

// --- Pybind11 module ---

PYBIND11_MODULE(_comms_mscclpp, m) {
  m.doc() = "MSCCL++ backend for TorchComm";

  py::class_<TorchCommMSCCLPP, std::shared_ptr<TorchCommMSCCLPP>>(m, "TorchCommMSCCLPP").def(py::init<>());
}
