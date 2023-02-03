#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <stdarg.h>
#include "debug.h"

using namespace std;

int mscclDebugLevel = -1;

void mscclppDebugInit()
{
  int lev = -1;
  const char *mscclpp_debug = getenv("MSCCLPP_DEBUG");
  if (mscclpp_debug == nullptr) {
    lev = MSCCLPP_LOG_NONE;
  } else {
    string mscclpp_debug_str(mscclpp_debug);
    if (mscclpp_debug_str == "INFO") {
      lev = MSCCLPP_LOG_INFO;
    } else if (mscclpp_debug_str == "DEBUG") {
      lev = MSCCLPP_LOG_DEBUG;
    } else if (mscclpp_debug_str == "ABORT") {
      lev = MSCCLPP_LOG_ABORT;
    } else {
      throw runtime_error("Unknown debug level given: " + mscclpp_debug_str);
    }
  }
  mscclDebugLevel = lev;
}

void mscclppDebugLog(mscclDebugLogLevel level, const char *filefunc, int line,
                     const char *fmt, ...)
{
  if (mscclDebugLevel == -1) {
    mscclppDebugInit();
  }
  if (level < mscclDebugLevel) {
    return;
  }
  string lev_str;
  if (level == MSCCLPP_LOG_INFO) {
    lev_str = "INFO";
  } else if (level == MSCCLPP_LOG_DEBUG) {
    lev_str = "DEBUG";
  } else if (level == MSCCLPP_LOG_ABORT) {
    lev_str = "ABORT";
  } else {
    assert(false);
  }
  char buffer[1024];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(buffer, 1024, fmt, vargs);
  va_end(vargs);
  stringstream ss;
  ss << "MSCCL " << lev_str << ": (" << filefunc << ":" << line << ") "
     << buffer << endl;
  cerr << ss.str();
}
