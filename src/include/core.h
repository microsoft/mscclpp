/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_CORE_H_
#define MSCCLPP_CORE_H_

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include <stdio.h>
#include <string.h>
#include "mscclpp.h"
#include "debug.h"
#include "alloc.h"
#include "param.h"

#ifdef PROFAPI
#define MSCCLPP_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define MSCCLPP_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

#endif // end include guard
