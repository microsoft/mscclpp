#include <gtest/gtest.h>

#include <mscclpp/utils.hpp>

TEST(UtilsTest, Timer) {
  mscclpp::Timer timer;
  sleep(1);
  int64_t elapsed = timer.elapsed();
  EXPECT_GE(elapsed, 1000000);

  timer.reset();
  sleep(1);
  elapsed = timer.elapsed();
  EXPECT_GE(elapsed, 1000000);
  EXPECT_LT(elapsed, 1100000);
}

TEST(UtilsTest, ScopedTimer) {
  mscclpp::ScopedTimer timerA("TimerA");
  mscclpp::ScopedTimer timerB("TimerB");
  sleep(1);
  int64_t elapsedA = timerA.timer.elapsed();
  int64_t elapsedB = timerB.timer.elapsed();
  EXPECT_GE(elapsedA, 1000000);
  EXPECT_GE(elapsedB, 1000000);
}

TEST(UtilsTest, getHostName) {
  std::string hostname = mscclpp::getHostName(1024, '.');
  EXPECT_FALSE(hostname.empty());
}
