#ifndef MSCCLPP_SPIN_HPP_
#define MSCCLPP_SPIN_HPP_

#ifdef __CUDACC__

#ifndef NDEBUG
#include <stdio.h>
#define MSCCLPP_SAFE_SPIN_PRINT(__cond)                         \
  do {                                                          \
    printf("mscclpp: spin is stuck. condition: " #__cond "\n"); \
  } while (0);                                                  \
#else  // NDEBUG
#define MSCCLPP_SAFE_SPIN_PRINT(__cond)
#endif  // NDEBUG

#define MSCCLPP_SAFE_SPIN_MAX_LOOPS 1000000000

// If a spin is stuck, escape from it and set status to 1.
#define MSCCLPP_SAFE_SPIN(__cond, __status)              \
  do {                                                   \
    uint64_t __spin_cnt = 0;                             \
    __status = 0;                                        \
    while (__cond) {                                     \
      if (__spin_cnt++ == MSCCLPP_SAFE_SPIN_MAX_LOOPS) { \
        MSCCLPP_SAFE_SPIN_PRINT(__cond);                 \
        __status = 1;                                    \
        break;                                           \
      }                                                  \
    }                                                    \
  } while (0);

// If a spin is stuck, print a warning and keep spinning.
#define MSCCLPP_SAFE_SPIN(__cond)                        \
  do {                                                   \
    uint64_t __spin_cnt = 0;                             \
    while (__cond) {                                     \
      if (__spin_cnt++ == MSCCLPP_SAFE_SPIN_MAX_LOOPS) { \
        MSCCLPP_SAFE_SPIN_PRINT(__cond);                 \
      }                                                  \
    }                                                    \
  } while (0);

#endif  // __CUDACC__

#endif  // MSCCLPP_SPIN_HPP_
