// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/errors.hpp>
#include <sstream>

#include "../framework.hpp"

TEST(ErrorsTest, OutputStream) {
  auto toString = [](mscclpp::ErrorCode errorCode) {
    std::ostringstream os;
    os << errorCode;
    return os.str();
  };

  EXPECT_EQ(toString(mscclpp::ErrorCode::SystemError), std::string("SystemError"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::InternalError), std::string("InternalError"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::RemoteError), std::string("RemoteError"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::InvalidUsage), std::string("InvalidUsage"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::Timeout), std::string("Timeout"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::Aborted), std::string("Aborted"));
  EXPECT_EQ(toString(mscclpp::ErrorCode::ExecutorError), std::string("ExecutorError"));
}

TEST(ErrorsTest, SystemError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: SystemError)"));
}

TEST(ErrorsTest, InternalError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: InternalError)"));
}

TEST(ErrorsTest, InvalidUsage) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: InvalidUsage)"));
}

TEST(ErrorsTest, Timeout) {
  mscclpp::Error error("test", mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.getErrorCode(), mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: Timeout)"));
}
