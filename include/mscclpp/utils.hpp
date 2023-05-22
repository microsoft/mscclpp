#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace mscclpp {

struct Timer {
  std::chrono::steady_clock::time_point start;

  Timer() { start = std::chrono::steady_clock::now(); }

  int64_t elapsed() {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

  void reset() { start = std::chrono::steady_clock::now(); }

  void print(const char* name) {
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("%s: %ld us\n", name, elapsed);
  }
};

struct ScopedTimer {
  Timer timer;
  const char* name;

  ScopedTimer(const char* name) : name(name) {}

  ~ScopedTimer() { timer.print(name); }
};

inline std::string getHostName(int maxlen, const char delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    std::strncpy(const_cast<char*>(hostname.data()), "unknown", maxlen);
    throw;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return hostname;
}

}  // namespace mscclpp

#endif  // MSCCLPP_UTILS_HPP_
