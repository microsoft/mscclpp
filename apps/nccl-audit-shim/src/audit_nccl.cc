// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlfcn.h>
#include <limits.h>
#include <link.h>

#include <cstring>
#include <filesystem>

static std::filesystem::path getLibDir() {
    Dl_info info{};
    if (dladdr((void*)&getLibDir, &info) == 0 || !info.dli_fname) {
        throw std::runtime_error("dladdr failed");
    }

    std::filesystem::path p(info.dli_fname);
    if (!p.is_absolute()) p = std::filesystem::absolute(p);
    return p.parent_path();
}

unsigned int la_version(unsigned int) { return LAV_CURRENT; }

char* la_objsearch(const char* name, uintptr_t*, unsigned int) {
  const char* library = "libmscclpp_nccl.so";
  if (strcmp(name, "libnccl.so.2") && strcmp(name, "libnccl.so") && strcmp(name, "librccl.so") &&
      strcmp(name, "librccl.so.1")) {
    return (char*)name;
  }
  std::string path = (getLibDir() / library).string();
  return strdup(path.c_str());
}