#include "errors.hpp"
#include "api.h"

namespace mscclpp {

BaseError::BaseError(std::string message, int errorCode) : std::runtime_error(message), errorCode_(errorCode)
{
}

int BaseError::getErrorCode() const
{
  return errorCode_;
}

MSCCLPP_API_CPP Error::Error(std::string message, ErrorCode errorCode) : BaseError(message, -1)
{
}

MSCCLPP_API_CPP CudaError::CudaError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

MSCCLPP_API_CPP CuError::CuError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

MSCCLPP_API_CPP IbError::IbError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

}; // namespace mscclpp
