// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/errors.hpp>
#include <mscclpp/utils.hpp>
#include <thread>

#include "../framework.hpp"

TEST(UtilsTest, getHostName) {
  std::string hostname1 = mscclpp::getHostName(1024, '.');
  EXPECT_FALSE(hostname1.empty());
  EXPECT_LE(hostname1.size(), 1024);

  EXPECT_EQ(mscclpp::getHostName(1024, hostname1[0]).size(), 0);

  std::string hostname2;

  std::thread th([&hostname2]() { hostname2 = mscclpp::getHostName(1024, '.'); });

  ASSERT_TRUE(th.joinable());
  th.join();

  EXPECT_EQ(hostname1, hostname2);
}
