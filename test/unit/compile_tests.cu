// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>

#undef NDEBUG
#ifndef DEBUG_BUILD
#define DEBUG_BUILD
#endif  // DEBUG_BUILD
#include <assert.h>

#include <mscclpp/poll_device.hpp>

TEST(CompileTest, Assert) { assert(true); }
