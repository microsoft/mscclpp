// aud_nccl.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void self_dir(char out[PATH_MAX]) {
  Dl_info di;
  if (dladdr((void*)&self_dir, &di) && di.dli_fname) {
    size_t n = strnlen(di.dli_fname, PATH_MAX - 1);
    char tmp[PATH_MAX];
    memcpy(tmp, di.dli_fname, n);
    tmp[n] = '\0';
    char* s = strrchr(tmp, '/');
    if (s)
      *s = '\0';
    else
      strcpy(tmp, ".");
    snprintf(out, PATH_MAX, "%s", tmp);
  } else
    snprintf(out, PATH_MAX, ".");
}
unsigned int la_version(unsigned int v) { return LAV_CURRENT; }
char* la_objsearch(const char* name, uintptr_t*, unsigned int) {
  if (strcmp(name, "libnccl.so.2") && strcmp(name, "libnccl.so")) return (char*)name;
  static char buf[PATH_MAX];
  char me[PATH_MAX];
  self_dir(me);
  snprintf(buf, sizeof(buf), "%s/libnccl.so.2", me);
  return buf;
}