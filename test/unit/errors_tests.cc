// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/errors.hpp>

#include "../framework.hpp"

// TODO: ErrorCode needs operator<< for EXPECT_EQ to work
// Using ASSERT_TRUE with manual comparisons as workaround

TEST(ErrorsTest, SystemError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::SystemError);
  ASSERT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::SystemError);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: SystemError)"));
}

TEST(ErrorsTest, InternalError) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InternalError);
  ASSERT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::InternalError);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: InternalError)"));
}

TEST(ErrorsTest, InvalidUsage) {
  mscclpp::Error error("test", mscclpp::ErrorCode::InvalidUsage);
  ASSERT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: InvalidUsage)"));
}

TEST(ErrorsTest, Timeout) {
  mscclpp::Error error("test", mscclpp::ErrorCode::Timeout);
  ASSERT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::Timeout);
  EXPECT_EQ(error.what(), std::string("test (mscclpp failure: Timeout)"));
}
