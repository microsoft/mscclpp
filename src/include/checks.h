/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_CHECKS_H_
#define MSCCLPP_CHECKS_H_

#include <cuda_runtime.h>

#include "debug.h"

// Check CUDA RT calls
#define CUDACHECK(cmd)                                    \
  do {                                                    \
    cudaError_t err = cmd;                                \
    if (err != cudaSuccess) {                             \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      return mscclppUnhandledCudaError;                   \
    }                                                     \
  } while (false)

#define CUDACHECKNORET(cmd)                               \
  do {                                                    \
    cudaError_t err = cmd;                                \
    if (err != cudaSuccess) {                             \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      return;                                             \
    }                                                     \
  } while (false)

#define CUDACHECKGOTO(cmd, res, label)                    \
  do {                                                    \
    cudaError_t err = cmd;                                \
    if (err != cudaSuccess) {                             \
      WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
      res = mscclppUnhandledCudaError;                    \
      goto label;                                         \
    }                                                     \
  } while (false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd)                                                                     \
  do {                                                                                           \
    cudaError_t err = cmd;                                                                       \
    if (err != cudaSuccess) {                                                                    \
      INFO(MSCCLPP_ALL, "%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
      (void)cudaGetLastError();                                                                  \
    }                                                                                            \
  } while (false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name)         \
  do {                               \
    int retval;                      \
    SYSCHECKVAL(call, name, retval); \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                      \
  do {                                                       \
    SYSCHECKSYNC(call, name, retval);                        \
    if (retval == -1) {                                      \
      WARN("Call to " name " failed : %s", strerror(errno)); \
      return mscclppSystemError;                             \
    }                                                        \
  } while (false)

#define SYSCHECKSYNC(call, name, retval)                                               \
  do {                                                                                 \
    retval = call;                                                                     \
    if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
      INFO(MSCCLPP_ALL, "Call to " name " returned %s, retrying", strerror(errno));    \
    } else {                                                                           \
      break;                                                                           \
    }                                                                                  \
  } while (true)

#define SYSCHECKGOTO(statement, res, label)                      \
  do {                                                           \
    if ((statement) == -1) {                                     \
      /* Print the back trace*/                                  \
      res = mscclppSystemError;                                  \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                \
    }                                                            \
  } while (0);

#define NEQCHECK(statement, value)                                              \
  do {                                                                          \
    if ((statement) != value) {                                                 \
      /* Print the back trace*/                                                 \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, mscclppSystemError); \
      return mscclppSystemError;                                                \
    }                                                                           \
  } while (0);

#define NEQCHECKGOTO(statement, value, res, label)               \
  do {                                                           \
    if ((statement) != value) {                                  \
      /* Print the back trace*/                                  \
      res = mscclppSystemError;                                  \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                \
    }                                                            \
  } while (0);

#define EQCHECK(statement, value)                                               \
  do {                                                                          \
    if ((statement) == value) {                                                 \
      /* Print the back trace*/                                                 \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, mscclppSystemError); \
      return mscclppSystemError;                                                \
    }                                                                           \
  } while (0);

#define EQCHECKGOTO(statement, value, res, label)                \
  do {                                                           \
    if ((statement) == value) {                                  \
      /* Print the back trace*/                                  \
      res = mscclppSystemError;                                  \
      INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                \
    }                                                            \
  } while (0);

// Propagate errors up
#define MSCCLPPCHECK(call)                                                                    \
  do {                                                                                        \
    mscclppResult_t res = call;                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      /* Print the back trace*/                                                               \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      return res;                                                                             \
    }                                                                                         \
  } while (0);

#define MSCCLPPCHECKGOTO(call, res, label)                                                    \
  do {                                                                                        \
    res = call;                                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      /* Print the back trace*/                                                               \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                                             \
    }                                                                                         \
  } while (0);

#define MSCCLPPWAIT(call, cond, abortFlagPtr)                                                 \
  do {                                                                                        \
    volatile uint32_t* tmpAbortFlag = (abortFlagPtr);                                         \
    mscclppResult_t res = call;                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      return mscclppInternalError;                                                            \
    }                                                                                         \
    if (tmpAbortFlag) NEQCHECK(*tmpAbortFlag, 0);                                             \
  } while (!(cond));

#define MSCCLPPWAITGOTO(call, cond, abortFlagPtr, res, label)                                 \
  do {                                                                                        \
    volatile uint32_t* tmpAbortFlag = (abortFlagPtr);                                         \
    res = call;                                                                               \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                  \
      if (mscclppDebugNoWarn == 0) INFO(MSCCLPP_ALL, "%s:%d -> %d", __FILE__, __LINE__, res); \
      goto label;                                                                             \
    }                                                                                         \
    if (tmpAbortFlag) NEQCHECKGOTO(*tmpAbortFlag, 0, res, label);                             \
  } while (!(cond));

#define MSCCLPPCHECKTHREAD(a, args)                                                      \
  do {                                                                                   \
    if (((args)->ret = (a)) != mscclppSuccess && (args)->ret != mscclppInProgress) {     \
      INFO(MSCCLPP_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__, (args)->ret); \
      return args;                                                                       \
    }                                                                                    \
  } while (0)

#define CUDACHECKTHREAD(a)                                                             \
  do {                                                                                 \
    if ((a) != cudaSuccess) {                                                          \
      INFO(MSCCLPP_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
      args->ret = mscclppUnhandledCudaError;                                           \
      return args;                                                                     \
    }                                                                                  \
  } while (0)

#endif
