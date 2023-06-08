#ifndef MSCCLPP_POLL_HPP_
#define MSCCLPP_POLL_HPP_

#ifdef __CUDACC__

#ifndef NDEBUG
#include <stdio.h>
#define POLL_PRINT_ON_STUCK(__cond)                             \
  do {                                                          \
    printf("mscclpp: spin is stuck. condition: " #__cond "\n"); \
  } while (0);
#else  // NDEBUG
#define POLL_PRINT_ON_STUCK(__cond)
#endif  // NDEBUG

// If a spin is stuck, escape from it and set status to 1.
#define POLL_MAYBE_JAILBREAK_ESCAPE(__cond, __max_spin_cnt, __status) \
  do {                                                                \
    uint64_t __spin_cnt = 0;                                          \
    __status = 0;                                                     \
    while (__cond) {                                                  \
      if (__spin_cnt++ == __max_spin_cnt) {                           \
        POLL_PRINT_ON_STUCK(__cond);                                  \
        __status = 1;                                                 \
        break;                                                        \
      }                                                               \
    }                                                                 \
  } while (0);

// If a spin is stuck, print a warning and keep spinning.
#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt) \
  do {                                               \
    uint64_t __spin_cnt = 0;                         \
    while (__cond) {                                 \
      if (__spin_cnt++ == __max_spin_cnt) {          \
        POLL_PRINT_ON_STUCK(__cond);                 \
      }                                              \
    }                                                \
  } while (0);

#endif  // __CUDACC__

#endif  // MSCCLPP_POLL_HPP_
