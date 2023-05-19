#ifndef MSCCLPP_CHECKS_OLD_HPP_
#define MSCCLPP_CHECKS_OLD_HPP_

#include <mscclpp/checks.hpp>

#define MSCCLPPTHROW(call)                                                                                        \
  do {                                                                                                            \
    mscclppResult_t res = call;                                                                                   \
    mscclpp::ErrorCode err = mscclpp::ErrorCode::InternalError;                                                   \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                      \
      if (res == mscclppInvalidUsage) {                                                                           \
        err = mscclpp::ErrorCode::InvalidUsage;                                                                   \
      } else if (res == mscclppSystemError) {                                                                     \
        err = mscclpp::ErrorCode::SystemError;                                                                    \
      }                                                                                                           \
      throw mscclpp::Error(std::string("Call to " #call " failed. ") + __FILE__ + ":" + std::to_string(__LINE__), \
                           err);                                                                                  \
    }                                                                                                             \
  } while (false)

#endif
