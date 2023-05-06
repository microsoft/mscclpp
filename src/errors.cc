#include "errors.hpp"

namespace mscclpp {

BaseError::BaseError(std::string message, int errorCode) : std::runtime_error(message), errorCode_(errorCode)
{
}

int BaseError::getErrorCode() const
{
  return errorCode_;
}

Error::Error(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

CudaError::CudaError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

CuError::CuError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

IbError::IbError(std::string message, int errorCode) : BaseError(message, errorCode)
{
}

}; // namespace mscclpp
