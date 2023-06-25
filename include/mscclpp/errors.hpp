// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ERRORS_HPP_
#define MSCCLPP_ERRORS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace mscclpp {

enum class ErrorCode {
  SystemError,
  InternalError,
  RemoteError,
  InvalidUsage,
  Timeout,
  Aborted,
};

std::string errorToString(enum ErrorCode error);

class BaseError : public std::runtime_error {
 public:
  BaseError(const std::string& message, int errorCode);
  explicit BaseError(int errorCode);
  virtual ~BaseError() = default;
  int getErrorCode() const;
  const char* what() const noexcept override;

 protected:
  std::string message_;
  int errorCode_;
};

class Error : public BaseError {
 public:
  Error(const std::string& message, ErrorCode errorCode);
  virtual ~Error() = default;
  ErrorCode getErrorCode() const;
};

class SysError : public BaseError {
 public:
  SysError(const std::string& message, int errorCode);
  virtual ~SysError() = default;
};

class CudaError : public BaseError {
 public:
  CudaError(const std::string& message, cudaError_t errorCode);
  virtual ~CudaError() = default;
};

class CuError : public BaseError {
 public:
  CuError(const std::string& message, CUresult errorCode);
  virtual ~CuError() = default;
};

class IbError : public BaseError {
 public:
  IbError(const std::string& message, int errorCode);
  virtual ~IbError() = default;
};

};  // namespace mscclpp

#endif  // MSCCLPP_ERRORS_HPP_
