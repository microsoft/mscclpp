#ifndef MSCCLPP_ERRORS_HPP_
#define MSCCLPP_ERRORS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace mscclpp {

enum class ErrorCode {
  SystemError,
  InternalError,
  InvalidUsage,
};

std::string errorToString(enum ErrorCode error);

class BaseError : public std::runtime_error {
 public:
  BaseError(std::string message, int errorCode);
  explicit BaseError(int errorCode);
  virtual ~BaseError() = default;
  int getErrorCode() const;
  const char* what() const noexcept override;

 private:
  int errorCode_;

 protected:
  std::string message_;
};

class Error : public BaseError {
 public:
  Error(std::string message, ErrorCode errorCode);
  virtual ~Error() = default;
};

class CudaError : public BaseError {
 public:
  CudaError(std::string message, cudaError_t errorCode);
  virtual ~CudaError() = default;
};

class CuError : public BaseError {
 public:
  CuError(std::string message, CUresult errorCode);
  virtual ~CuError() = default;
};

class IbError : public BaseError {
 public:
  IbError(std::string message, int errorCode);
  virtual ~IbError() = default;
};

};  // namespace mscclpp

#endif  // MSCCLPP_ERRORS_HPP_
