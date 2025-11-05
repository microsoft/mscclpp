// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ibverbs_wrapper.hpp"

#include <dlfcn.h>

#include <memory>

#include "logger.hpp"

namespace mscclpp {

static std::unique_ptr<void, int (*)(void*)> globalIBVerbsHandle(nullptr, &::dlclose);

void* IBVerbs::dlsym(const std::string& symbol) {
  if (!globalIBVerbsHandle) {
    const char* possibleLibNames[] = {"libibverbs.so", "libibverbs.so.1", nullptr};
    for (int i = 0; possibleLibNames[i] != nullptr; i++) {
      void* handle = ::dlopen(possibleLibNames[i], RTLD_NOW);
      if (handle) {
        globalIBVerbsHandle.reset(handle);
        break;
      }
    }
    if (!globalIBVerbsHandle) {
      THROW(NET, SysError, errno, "Failed to open libibverbs: ", std::string(::dlerror()));
    }
  }
  void* ptr = ::dlsym(globalIBVerbsHandle.get(), symbol.c_str());
  if (!ptr) {
    THROW(NET, SysError, errno, "Failed to load libibverbs symbol: ", symbol);
  }
  return ptr;
}

}  // namespace mscclpp
