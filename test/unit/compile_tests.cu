// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../framework.hpp"

#undef NDEBUG
#ifndef DEBUG_BUILD
#define DEBUG_BUILD
#endif  // DEBUG_BUILD
#include <assert.h>

#include <mscclpp/poll_device.hpp>

TEST(CompileTest, Assert) { assert(true); }
