// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "constants.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[1];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

// HT uses the rank index as a named-barrier ID and dispatch assigns one warp
// per rank, so 16 is the architectural maximum for this launch family.
#define SWITCH_RANKS(case_macro)                       \
  do {                                                 \
    switch (num_ranks) {                               \
      case 2:                                          \
        case_macro(2);                                 \
      case 4:                                          \
        case_macro(4);                                 \
      case 8:                                          \
        case_macro(8);                                 \
      case 16:                                         \
        case_macro(16);                                \
      default:                                         \
        EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                  \
  } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)    \
  do {                                                \
    switch (num_ranks) {                              \
      case 2:                                         \
        case_macro(dtype, 2);                         \
      case 4:                                         \
        case_macro(dtype, 4);                         \
      case 8:                                         \
        case_macro(dtype, 8);                         \
      case 16:                                        \
        case_macro(dtype, 16);                        \
      default:                                        \
        EP_HOST_ASSERT(false && "Unsupported ranks"); \
    }                                                 \
  } while (false)

#define SWITCH_TYPES(case_macro)                     \
  do {                                               \
    switch (type) {                                  \
      case CUDA_R_16BF:                              \
        case_macro(nv_bfloat16);                     \
      case CUDA_R_32F:                               \
        case_macro(float);                           \
      default:                                       \
        EP_HOST_ASSERT(false && "Unsupported type"); \
    }                                                \
  } while (false)
