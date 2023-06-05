#include <gtest/gtest.h>
#include <mscclpp/errors.hpp>

TEST(ErrorsTest, SystemError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.getErrorCode(), static_cast<int>(mscclpp::ErrorCode::SystemError));
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: SystemError)"));
}

TEST(ErrorsTest, InternalError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.getErrorCode(), static_cast<int>(mscclpp::ErrorCode::InternalError));
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: InternalError)"));
}

TEST(ErrorsTest, InvalidUsage) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.getErrorCode(), static_cast<int>(mscclpp::ErrorCode::InvalidUsage));
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: InvalidUsage)"));
}

TEST(ErrorsTest, Timeout) {
  mscclpp::Error error("test", mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.getErrorCode(), static_cast<int>(mscclpp::ErrorCode::Timeout));
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: Timeout)"));
}

TEST(ErrorsTest, UnknownError) {
  mscclpp::Error error("test", static_cast<mscclpp::ErrorCode>(-1));
  EXPECT_EQ(error.getErrorCode(), -1);
  EXPECT_EQ(error.what(), std::string("test (Mscclpp failure: UnknownError)"));
}
