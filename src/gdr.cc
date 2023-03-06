#include "gdr.h"

// Used to make the GDR library calls thread safe
pthread_mutex_t gdrLock = PTHREAD_MUTEX_INITIALIZER;

gdr_t wrap_gdr_open(void) {
  return gdr_open();
}

mscclppResult_t wrap_gdr_close(gdr_t g) {
  int ret = gdr_close(g);
  if (ret != 0) {
    WARN("gdr_close() failed: %d", ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) {
  int ret;
  GDRLOCKCALL(gdr_pin_buffer(g, addr, size, p2p_token, va_space, handle), ret);
  if (ret != 0) {
    WARN("gdr_pin_buffer(addr %lx, size %zi) failed: %d", addr, size, ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  int ret;
  GDRLOCKCALL(gdr_unpin_buffer(g, handle), ret);
  if (ret != 0) {
    WARN("gdr_unpin_buffer(handle %lx) failed: %d", handle.h, ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info) {
  int ret;
  GDRLOCKCALL(gdr_get_info(g, handle, info), ret);
  if (ret != 0) {
    WARN("gdr_get_info(handle %lx) failed: %d", handle.h, ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size) {
  int ret;
  GDRLOCKCALL(gdr_map(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_map(handle %lx, size %zi) failed: %d", handle.h, size, ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}

mscclppResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size) {
  int ret;
  GDRLOCKCALL(gdr_unmap(g, handle, va, size), ret);
  if (ret != 0) {
    WARN("gdr_unmap(handle %lx, va %p, size %zi) failed: %d", handle.h, va, size, ret);
    return mscclppSystemError;
  }
  return mscclppSuccess;
}
