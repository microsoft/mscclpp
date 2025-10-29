// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <mscclpp/errors.hpp>

namespace nb = nanobind;
using namespace mscclpp;

#define REGISTER_EXCEPTION_TRANSLATOR(name_)                                                                         \
  nb::register_exception_translator(                                                                                 \
      [](const std::exception_ptr &p, void *payload) {                                                               \
        try {                                                                                                        \
          std::rethrow_exception(p);                                                                                 \
        } catch (const name_ &e) {                                                                                   \
          PyErr_SetObject(reinterpret_cast<PyObject *>(payload),                                                     \
                          PyTuple_Pack(2, PyLong_FromLong(long(e.getErrorCode())), PyUnicode_FromString(e.what()))); \
        }                                                                                                            \
      },                                                                                                             \
      m.attr(#name_).ptr());

void register_error(nb::module_ &m) {
  nb::enum_<ErrorCode>(m, "ErrorCode")
      .value("SystemError", ErrorCode::SystemError)
      .value("InternalError", ErrorCode::InternalError)
      .value("RemoteError", ErrorCode::RemoteError)
      .value("InvalidUsage", ErrorCode::InvalidUsage)
      .value("Timeout", ErrorCode::Timeout)
      .value("Aborted", ErrorCode::Aborted)
      .value("ExecutorError", ErrorCode::ExecutorError);

  nb::exception<BaseError>(m, "BaseError");
  REGISTER_EXCEPTION_TRANSLATOR(BaseError);

  nb::exception<Error>(m, "Error", m.attr("BaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(Error);

  nb::exception<SysError>(m, "SysError", m.attr("BaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(SysError);

  nb::exception<CudaError>(m, "CudaError", m.attr("BaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(CudaError);

  nb::exception<CuError>(m, "CuError", m.attr("BaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(CuError);

  nb::exception<IbError>(m, "IbError", m.attr("BaseError").ptr());
  REGISTER_EXCEPTION_TRANSLATOR(IbError);
}
