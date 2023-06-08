#include <gtest/gtest.h>

#include <mscclpp/errors.hpp>
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

TEST(UtilsTest, TimerTimeout) {
  mscclpp::Timer timer(1);
  ASSERT_THROW(sleep(2), mscclpp::Error);
}

TEST(UtilsTest, TimerTimeoutReset) {
  mscclpp::Timer timer(3);
  sleep(2);
  // Resetting the timer should prevent the timeout.
  timer.reset();
  ASSERT_NO_THROW(sleep(2));

  // Elapsed time should be slightly larger than 2 seconds.
  EXPECT_GT(timer.elapsed(), 2000000);
  EXPECT_LT(timer.elapsed(), 2100000);
}

TEST(UtilsTest, ScopedTimer) {
  mscclpp::ScopedTimer timerA("UtilsTest.ScopedTimer.A");
  mscclpp::ScopedTimer timerB("UtilsTest.ScopedTimer.B");
  sleep(1);
  int64_t elapsedA = timerA.elapsed();
  int64_t elapsedB = timerB.elapsed();
  EXPECT_GE(elapsedA, 1000000);
  EXPECT_GE(elapsedB, 1000000);
}

TEST(UtilsTest, getHostName) {
  std::string hostname = mscclpp::getHostName(1024, '.');
  EXPECT_FALSE(hostname.empty());
}
