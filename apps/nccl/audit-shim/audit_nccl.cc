// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <limits.h>
#include <link.h>

#include <cstring>

extern "C" __attribute__((visibility("default"))) unsigned int la_version(unsigned int) { return LAV_CURRENT; }

extern "C" __attribute__((visibility("default"))) char *la_objsearch(const char *name, uintptr_t *, unsigned int) {
  const char *library = "libmscclpp_nccl.so";
  if (strcmp(name, "libnccl.so.2") && strcmp(name, "libnccl.so") && strcmp(name, "librccl.so") &&
      strcmp(name, "librccl.so.1")) {
    return (char *)name;
  }
  return (char *)library;
}