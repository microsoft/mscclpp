// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/errors.hpp>

namespace nb = nanobind;
using namespace mscclpp;

#define REGISTER_EXCEPTION_TRANSLATOR(name_, py_name_)                                                                 \
  nb::register_exception_translator(                                                                                 \
      [](const std::exception_ptr &p, void *payload) {                                                               \
        try {                                                                                                        \
          std::rethrow_exception(p);                                                                                 \
        } catch (const name_ &e) {                                                                                   \
          PyErr_SetObject(reinterpret_cast<PyObject *>(payload),                                                     \
                          PyTuple_Pack(2, PyLong_FromLong(long(e.getErrorCode())), PyUnicode_FromString(e.what()))); \
        }                                                                                                            \
      },                                                                                                             \
      m.attr(py_name_).ptr());

void register_error(nb::module_ &m) {
  nb::enum_<ErrorCode>(m, "CppErrorCode")
      .value("SystemError", ErrorCode::SystemError)
      .value("InternalError", ErrorCode::InternalError)
      .value("RemoteError", ErrorCode::RemoteError)
      .value("InvalidUsage", ErrorCode::InvalidUsage)
      .value("Timeout", ErrorCode::Timeout)
      .value("Aborted", ErrorCode::Aborted)
      .value("ExecutorError", ErrorCode::ExecutorError);

  nb::exception<BaseError>(m, "CppBaseError");
  REGISTER_EXCEPTION_TRANSLATOR(BaseError, "CppBaseError");

  nb::exception<Error>(m, "CppError", m.attr("CppBaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(Error, "CppError");

  nb::exception<SysError>(m, "CppSysError", m.attr("CppBaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(SysError, "CppSysError");

  nb::exception<CudaError>(m, "CppCudaError", m.attr("CppBaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(CudaError, "CppCudaError");

  nb::exception<CuError>(m, "CppCuError", m.attr("CppBaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(CuError, "CppCuError");

  nb::exception<IbError>(m, "CppIbError", m.attr("CppBaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(IbError, "CppIbError");
}
