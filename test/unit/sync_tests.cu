/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "sync_tests.hpp"

TEST_F(DeviceSyncerTestFixture, execute_1_1) {
  execute(1, 1);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_1) {
  execute(1024, 1);
}

TEST_F(DeviceSyncerTestFixture, execute_1_2) {
  execute(1, 2);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_2) {
  execute(1024, 2);
}

TEST_F(DeviceSyncerTestFixture, execute_1_4) {
  execute(1, 4);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_4) {
  execute(1024, 4);
}

TEST_F(DeviceSyncerTestFixture, execute_1_8) {
  execute(1, 8);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_8) {
  execute(1024, 8);
}

TEST_F(DeviceSyncerTestFixture, execute_1_16) {
  execute(1, 16);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_16) {
  execute(1024, 16);
}

TEST_F(DeviceSyncerTestFixture, execute_1_32) {
  execute(1, 32);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_32) {
  execute(1024, 32);
}

TEST_F(DeviceSyncerTestFixture, execute_1_64) {
  execute(1, 64);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_64) {
  execute(1024, 64);
}

TEST_F(DeviceSyncerTestFixture, execute_1_104) {
  execute(1, 104);
}

TEST_F(DeviceSyncerTestFixture, execute_1024_104) {
  execute(1024, 104);
}

TEST_F(DeviceSyncerTestFixture, execute_1_128) {
  execute(1, 128);
}

TEST_F(DeviceSyncerTestFixture, execute_1_256) {
  execute(1, 256);
}

TEST_F(DeviceSyncerTestFixture, execute_1_512) {
  execute(1, 512);
}

TEST_F(DeviceSyncerTestFixture, execute_1_1024) {
  execute(1, 1024);
}
