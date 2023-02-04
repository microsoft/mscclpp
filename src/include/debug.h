#ifndef MSCCLPP_DEBUG_H_
#define MSCCLPP_DEBUG_H_

extern int mscclDebugLevel;

typedef enum {
  MSCCLPP_LOG_NONE = 0,
  MSCCLPP_LOG_WARN = 1,
  MSCCLPP_LOG_INFO = 2,
  MSCCLPP_LOG_DEBUG = 3,
  MSCCLPP_LOG_ABORT = 4,
} mscclDebugLogLevel;

void mscclppDebugLog(mscclDebugLogLevel level, const char *filefunc, int line,
                     const char *fmt, ...);

#define INFO(...) mscclppDebugLog(MSCCLPP_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...) mscclppDebugLog(MSCCLPP_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define DEBUG(...) mscclppDebugLog(MSCCLPP_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define ABORT(...) mscclppDebugLog(MSCCLPP_LOG_ABORT, __FILE__, __LINE__, __VA_ARGS__)

#define MSCCLPPCHECK(call) do { \
  mscclppResult_t res = call; \
  if (res != mscclppSuccess) { \
    /* Print the back trace*/ \
    INFO("%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return mscclppSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO("Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#endif // MSCCLPP_DEBUG_H_
