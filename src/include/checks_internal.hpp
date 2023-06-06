#ifndef MSCCLPP_CHECKS_OLD_HPP_
#define MSCCLPP_CHECKS_OLD_HPP_

#include <mscclpp/checks.hpp>

#include "debug.h"

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

// Check system calls
#define SYSCHECK(call, name)         \
  do {                               \
    int retval;                      \
    SYSCHECKVAL(call, name, retval); \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                      \
  do {                                                       \
    SYSCHECKSYNC(call, name, retval);                        \
    if (retval == -1) {                                      \
      WARN("Call to " name " failed : %s", strerror(errno)); \
      return mscclppSystemError;                             \
    }                                                        \
  } while (false)

#define SYSCHECKSYNC(call, name, retval)                                               \
  do {                                                                                 \
    retval = call;                                                                     \
    if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
      INFO(MSCCLPP_ALL, "Call to " name " returned %s, retrying", strerror(errno));    \
    } else {                                                                           \
      break;                                                                           \
    }                                                                                  \
  } while (true)

#define SYSCHECKGOTO(statement, res, label)                      \
  do {                                                           \
    if ((statement) == -1) {                                     \
      /* Print the back trace*/                                  \
      res = mscclppSystemError;                                  \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                \
    }                                                            \
  } while (0);

#define EQCHECK(statement, value)                                               \
  do {                                                                          \
    if ((statement) == value) {                                                 \
      /* Print the back trace*/                                                 \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, mscclppSystemError); \
      return mscclppSystemError;                                                \
    }                                                                           \
  } while (0);

#define EQCHECKGOTO(statement, value, res, label)                \
  do {                                                           \
    if ((statement) == value) {                                  \
      /* Print the back trace*/                                  \
      res = mscclppSystemError;                                  \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                \
    }                                                            \
  } while (0);

// Propagate errors up
#define MSCCLPPCHECK(call)                                                                    \
  do {                                                                                        \
    mscclppResult_t res = call;                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      /* Print the back trace*/                                                               \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      return res;                                                                             \
    }                                                                                         \
  } while (0);

#define MSCCLPPCHECKGOTO(call, res, label)                                                    \
  do {                                                                                        \
    res = call;                                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      /* Print the back trace*/                                                               \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                                             \
    }                                                                                         \
  } while (0);

#endif
