#ifndef MSCCLPP_GDR_H_
#define MSCCLPP_GDR_H_

#include "gdrapi.h"
#include "debug.h"
#include "checks.h"
#include "align.h"
#include "alloc.h"

// These can be used if the GDR library isn't thread safe
#include <pthread.h>
extern pthread_mutex_t gdrLock;
#define GDRLOCK() pthread_mutex_lock(&gdrLock)
#define GDRUNLOCK() pthread_mutex_unlock(&gdrLock)
#define GDRLOCKCALL(cmd, ret) do {                      \
    GDRLOCK();                                          \
    ret = cmd;                                          \
    GDRUNLOCK();                                        \
} while(false)

#define GDRCHECK(cmd) do {                              \
    int e;                                              \
    /* GDRLOCKCALL(cmd, e); */                          \
    e = cmd;                                            \
    if( e != 0 ) {                                      \
      WARN("GDRCOPY failure %d", e);                    \
      return mscclppSystemError;                        \
    }                                                   \
} while(false)

gdr_t wrap_gdr_open(void);
mscclppResult_t wrap_gdr_close(gdr_t g);
mscclppResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
mscclppResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
mscclppResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
mscclppResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
mscclppResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);

// Global GDR driver handle
extern gdr_t mscclppGdrCopy;

typedef struct gdr_mem_desc {
  void *gdrDevMem;
  void *gdrMap;
  size_t gdrOffset;
  size_t gdrMapSize;
  gdr_mh_t gdrMh;
} gdr_mem_desc_t;

static gdr_t mscclppGdrInit() {
  // int libMajor, libMinor, drvMajor, drvMinor;
  gdr_t handle = wrap_gdr_open();

  // if (handle != NULL) {
  //   mscclppResult_t res;

  //   // Query the version of libgdrapi
  //   MSCCLPPCHECKGOTO(wrap_gdr_runtime_get_version(&libMajor, &libMinor), res, error);

  //   // Query the version of gdrdrv driver
  //   MSCCLPPCHECKGOTO(wrap_gdr_driver_get_version(handle, &drvMajor, &drvMinor), res, error);

  //   // Only support GDRAPI 2.1 and later
  //   if (libMajor < 2 || (libMajor == 2 && libMinor < 1) || drvMajor < 2 || (drvMajor == 2 && drvMinor < 1)) {
  //     goto error;
  //   }
  //   else
  //     INFO(MSCCLPP_INIT, "GDRCOPY enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
  // }
  return handle;
// error:
//   if (handle != NULL) (void) wrap_gdr_close(handle);
//   return NULL;
}

template <typename T>
mscclppResult_t mscclppGdrCudaCallocDebug(T** ptr, T** devPtr, size_t nelem, void** gdrDesc, const char *filefunc, int line) {
  mscclppResult_t result = mscclppSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  *devPtr = nullptr;
  *gdrDesc = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  gdr_info_t info;
  size_t mapSize;
  gdr_mh_t mh;
  char *devMem;
  void *gdrMap;
  ssize_t off;
  gdr_mem_desc_t* md;
  uint64_t alignedAddr;
  size_t align;

  mapSize = sizeof(T)*nelem;

  // GDRCOPY Pinned buffer has to be a minimum of a GPU_PAGE_SIZE
  ALIGN_SIZE(mapSize, GPU_PAGE_SIZE);
  // GDRCOPY Pinned buffer has to be GPU_PAGE_SIZE aligned too
  MSCCLPPCHECKGOTO(mscclppCudaCalloc(&devMem, mapSize+GPU_PAGE_SIZE-1), result, finish);
  alignedAddr = (((uint64_t) devMem) + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  align = alignedAddr - (uint64_t)devMem;
  WARN("GDR: mscclppGdrCopy %p alignedAddr %p, mapSize %lu", mscclppGdrCopy, (void*)alignedAddr, mapSize);
  MSCCLPPCHECKGOTO(wrap_gdr_pin_buffer(mscclppGdrCopy, alignedAddr, mapSize, 0, 0, &mh), result, finish);

  MSCCLPPCHECKGOTO(wrap_gdr_map(mscclppGdrCopy, mh, &gdrMap, mapSize), result, finish);

  MSCCLPPCHECKGOTO(wrap_gdr_get_info(mscclppGdrCopy, mh, &info), result, finish);

  // Will offset ever be non zero ?
  off = info.va - alignedAddr;

  MSCCLPPCHECKGOTO(mscclppCalloc(&md, 1), result, finish);
  md->gdrDevMem = devMem;
  md->gdrMap = gdrMap;
  md->gdrMapSize = mapSize;
  md->gdrOffset = off+align;
  md->gdrMh = mh;
  *gdrDesc = md;

  *ptr = (T *)((char *)gdrMap+off);
  if (devPtr) *devPtr = (T *)(devMem+off+align);

  TRACE(mscclpp_INIT, "GDRCOPY : allocated devMem %p gdrMap %p offset %lx mh %lx mapSize %zi at %p",
       md->gdrDevMem, md->gdrMap, md->gdrOffset, md->gdrMh.h, md->gdrMapSize, *ptr);

  return mscclppSuccess;

finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) WARN("Failed to CUDA calloc %ld bytes", nelem*sizeof(T));
  INFO(MSCCLPP_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define mscclppGdrCudaCalloc(...) mscclppGdrCudaCallocDebug(__VA_ARGS__, __FILE__, __LINE__)


static mscclppResult_t mscclppGdrCudaFree(void* gdrDesc) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrDesc;
  MSCCLPPCHECK(wrap_gdr_unmap(mscclppGdrCopy, md->gdrMh, md->gdrMap, md->gdrMapSize));
  MSCCLPPCHECK(wrap_gdr_unpin_buffer(mscclppGdrCopy, md->gdrMh));
  CUDACHECK(cudaFree(md->gdrDevMem));
  free(md);

  return mscclppSuccess;
}


#endif
