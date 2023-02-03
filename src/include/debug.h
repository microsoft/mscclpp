#ifndef MSCCLPP_DEBUG_H_
#define MSCCLPP_DEBUG_H_

extern int mscclDebugLevel;

typedef enum {
  MSCCLPP_LOG_NONE = 0,
  MSCCLPP_LOG_INFO = 1,
  MSCCLPP_LOG_DEBUG = 2,
  MSCCLPP_LOG_ABORT = 3,
} mscclDebugLogLevel;

void mscclppDebugLog(mscclDebugLogLevel level, const char *filefunc, int line,
                     const char *fmt, ...);

#define INFO(...) mscclppDebugLog(MSCCLPP_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define DEBUG(...) mscclppDebugLog(MSCCLPP_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define ABORT(...) mscclppDebugLog(MSCCLPP_LOG_ABORT, __FILE__, __LINE__, __VA_ARGS__)

#endif // MSCCLPP_DEBUG_H_
