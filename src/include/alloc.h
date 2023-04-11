/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_ALLOC_H_
#define MSCCLPP_ALLOC_H_

#include "align.h"
#include "checks.h"
#include "mscclpp.h"
#include "utils.h"
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

template <typename T> mscclppResult_t mscclppCudaHostCallocDebug(T** ptr, size_t nelem, const char* filefunc, int line)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaHostAlloc(ptr, nelem * sizeof(T), cudaHostAllocMapped | cudaHostAllocWriteCombined), result,
                finish);
  memset(*ptr, 0, nelem * sizeof(T));
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr)
    WARN("Failed to CUDA host alloc %ld bytes", nelem * sizeof(T));
  INFO(MSCCLPP_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem * sizeof(T), *ptr);
  return result;
}
#define mscclppCudaHostCalloc(...) mscclppCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

inline mscclppResult_t mscclppCudaHostFree(void* ptr)
{
  CUDACHECK(cudaFreeHost(ptr));
  return mscclppSuccess;
}

template <typename T> mscclppResult_t mscclppCallocDebug(T** ptr, size_t nelem, const char* filefunc, int line)
{
  void* p = malloc(nelem * sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * sizeof(T));
    return mscclppSystemError;
  }
  INFO(MSCCLPP_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem * sizeof(T), p);
  memset(p, 0, nelem * sizeof(T));
  *ptr = (T*)p;
  return mscclppSuccess;
}
#define mscclppCalloc(...) mscclppCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T> mscclppResult_t mscclppRealloc(T** ptr, size_t oldNelem, size_t nelem)
{
  if (nelem < oldNelem)
    return mscclppInternalError;
  if (nelem == oldNelem)
    return mscclppSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem * sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * sizeof(T));
    return mscclppSystemError;
  }
  memcpy(p, oldp, oldNelem * sizeof(T));
  free(oldp);
  memset(p + oldNelem, 0, (nelem - oldNelem) * sizeof(T));
  *ptr = (T*)p;
  INFO(MSCCLPP_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem * sizeof(T), nelem * sizeof(T),
       *ptr);
  return mscclppSuccess;
}

template <typename T> mscclppResult_t mscclppCudaMallocDebug(T** ptr, size_t nelem, const char* filefunc, int line)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMalloc(ptr, nelem * sizeof(T)), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr)
    WARN("Failed to CUDA malloc %ld bytes", nelem * sizeof(T));
  INFO(MSCCLPP_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem * sizeof(T), *ptr);
  return result;
}
#define mscclppCudaMalloc(...) mscclppCudaMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T> mscclppResult_t mscclppCudaCallocDebug(T** ptr, size_t nelem, const char* filefunc, int line)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECKGOTO(cudaMalloc(ptr, nelem * sizeof(T)), result, finish);
  CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem * sizeof(T), stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr)
    WARN("Failed to CUDA calloc %ld bytes", nelem * sizeof(T));
  INFO(MSCCLPP_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem * sizeof(T), *ptr);
  return result;
}
#define mscclppCudaCalloc(...) mscclppCudaCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
mscclppResult_t mscclppCudaCallocAsyncDebug(T** ptr, size_t nelem, cudaStream_t stream, const char* filefunc, int line)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMalloc(ptr, nelem * sizeof(T)), result, finish);
  CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem * sizeof(T), stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr)
    WARN("Failed to CUDA calloc async %ld bytes", nelem * sizeof(T));
  INFO(MSCCLPP_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem * sizeof(T), *ptr);
  return result;
}
#define mscclppCudaCallocAsync(...) mscclppCudaCallocAsyncDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T> mscclppResult_t mscclppCudaMemcpy(T* dst, T* src, size_t nelem)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  MSCCLPPCHECKGOTO(mscclppCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T> mscclppResult_t mscclppCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem * sizeof(T), cudaMemcpyDefault, stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T> mscclppResult_t mscclppCudaFree(T* ptr)
{
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaFree(ptr), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline mscclppResult_t mscclppIbMallocDebug(void** ptr, size_t size, const char* filefunc, int line)
{
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0)
    return mscclppSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(MSCCLPP_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return mscclppSuccess;
}
#define mscclppIbMalloc(...) mscclppIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
