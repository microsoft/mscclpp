#ifndef MSCCLPP_ERRORS_HPP_
#define MSCCLPP_ERRORS_HPP_

#include <stdexcept>

namespace mscclpp {

enum class ErrorCode {
  SystemError,
  InternalError,
  InvalidUsage,
};

class BaseError : public std::runtime_error {
 public:
  BaseError(std::string message, int errorCode);
  virtual ~BaseError() = default;
  int getErrorCode() const;

 private:
  int errorCode_;
};

class Error : public BaseError {
 public:
  Error(std::string message, ErrorCode errorCode);
  virtual ~Error() = default;
};

class CudaError : public BaseError {
 public:
  CudaError(std::string message, int errorCode);
  virtual ~CudaError() = default;
};

class CuError : public BaseError {
 public:
  CuError(std::string message, int errorCode);
  virtual ~CuError() = default;
};

class IbError : public BaseError {
 public:
  IbError(std::string message, int errorCode);
  virtual ~IbError() = default;
};

};  // namespace mscclpp

#endif  // MSCCLPP_ERRORS_HPP_
