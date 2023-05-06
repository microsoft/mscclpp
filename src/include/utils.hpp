#ifndef MSCCLPP_UTILS_HPP_
#define MSCCLPP_UTILS_HPP_

#include <chrono>
#include <stdio.h>

namespace mscclpp {

struct Timer
{
  std::chrono::steady_clock::time_point start;

  Timer()
  {
    start = std::chrono::steady_clock::now();
  }

  int64_t elapsed()
  {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

  void reset()
  {
    start = std::chrono::steady_clock::now();
  }

  void print(const char* name)
  {
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("%s: %ld us\n", name, elapsed);
  }
};

struct ScopedTimer
{
  Timer timer;
  const char* name;

  ScopedTimer(const char* name) : name(name)
  {
  }

  ~ScopedTimer()
  {
    timer.print(name);
  }
};

} // namespace mscclpp

#endif // MSCCLPP_UTILS_HPP_
