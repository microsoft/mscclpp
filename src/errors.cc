#include <cstring>
#include <mscclpp/errors.hpp>

#include "api.h"

namespace mscclpp {

std::string errorToString(enum ErrorCode error) {
  switch (error) {
    case ErrorCode::SystemError:
      return "SystemError";
    case ErrorCode::InternalError:
      return "InternalError";
    case ErrorCode::InvalidUsage:
      return "InvalidUsage";
    case ErrorCode::Timeout:
      return "Timeout";
    default:
      return "UnknownError";
  }
}

BaseError::BaseError(std::string message, int errorCode)
    : std::runtime_error(""), message_(message), errorCode_(errorCode) {}

BaseError::BaseError(int errorCode) : std::runtime_error(""), errorCode_(errorCode) {}

int BaseError::getErrorCode() const { return errorCode_; }

const char* BaseError::what() const noexcept { return message_.c_str(); }

MSCCLPP_API_CPP Error::Error(std::string message, ErrorCode errorCode) : BaseError(static_cast<int>(errorCode)) {
  message_ = message + " (Mscclpp failure: " + errorToString(errorCode) + ")";
}

MSCCLPP_API_CPP CudaError::CudaError(std::string message, cudaError_t errorCode) : BaseError(errorCode) {
  message_ = message + " (Cuda failure: " + cudaGetErrorString(errorCode) + ")";
}

MSCCLPP_API_CPP CuError::CuError(std::string message, CUresult errorCode) : BaseError(errorCode) {
  const char* errStr;
  cuGetErrorString(errorCode, &errStr);
  message_ = message + " (Cu failure: " + errStr + ")";
}

MSCCLPP_API_CPP IbError::IbError(std::string message, int errorCode) : BaseError(errorCode) {
  message_ = message + " (Ib failure: " + std::strerror(errorCode) + ")";
}

};  // namespace mscclpp
