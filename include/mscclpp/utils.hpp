#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <mscclpp/errors.hpp>
#include <sstream>
#include <string>

// Throw upon SIGALRM.
static void sigalrmTimeoutHandler(int) {
  signal(SIGALRM, SIG_IGN);
  throw mscclpp::Error("Timer timed out", mscclpp::ErrorCode::Timeout);
}

namespace mscclpp {

struct Timer {
  std::chrono::steady_clock::time_point start;
  const int timeout;

  Timer(int timeout = -1) : timeout(timeout) {
    if (timeout > 0) {
      signal(SIGALRM, sigalrmTimeoutHandler);
      alarm(timeout);
    }
    start = std::chrono::steady_clock::now();
  }

  ~Timer() {
    if (timeout > 0) {
      alarm(0);
      signal(SIGALRM, SIG_DFL);
    }
  }

  int64_t elapsed() {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

  void reset() {
    if (timeout > 0) {
      signal(SIGALRM, sigalrmTimeoutHandler);
      alarm(timeout);
    }
    start = std::chrono::steady_clock::now();
  }

  void print(const std::string& name) {
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::stringstream ss;
    ss << name << ": " << elapsed << " us\n";
    std::cout << ss.str();
  }
};

struct ScopedTimer {
  Timer timer;
  const std::string name;

  ScopedTimer(const std::string& name) : name(name) {}

  ~ScopedTimer() { timer.print(name); }
};

inline std::string getHostName(int maxlen, const char delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    std::strncpy(const_cast<char*>(hostname.data()), "unknown", maxlen);
    throw Error("gethostname failed", ErrorCode::SystemError);
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return hostname;
}

}  // namespace mscclpp

#endif  // MSCCLPP_UTILS_HPP_
