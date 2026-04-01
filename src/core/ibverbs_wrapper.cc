// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ibverbs_wrapper.hpp"

#include <dlfcn.h>

#include <memory>
#include <mscclpp/env.hpp>

#include "logger.hpp"

// NOTE: MRC_SUPPORT is a temporal macro that makes the current MRC implementation work.
// MRC_SUPPORT is needed because the current libibverbs implmentation of MRC does not provide
// all symbols that we need, so we need to load some symbols from the original libibverbs.
// This macro will be removed (set 0) once MRC provides all necessary symbols.
// Non-MRC environments will not be affected by this macro as long as VMRC_LIBIBVERBS_SO
// environment variable is not set.
#define MRC_SUPPORT 1
#if (MRC_SUPPORT)
#include <cstdlib>
#include <set>
#endif  // (MRC_SUPPORT)

namespace mscclpp {

static std::unique_ptr<void, int (*)(void*)> globalIBVerbsHandle(nullptr, &::dlclose);
#if (MRC_SUPPORT)
static std::unique_ptr<void, int (*)(void*)> globalOrigIBVerbsHandle(nullptr, &::dlclose);
#endif  // (MRC_SUPPORT)

void* IBVerbs::dlsym(const std::string& symbol, bool allowReturnNull) {
#if (MRC_SUPPORT)
  static std::set<std::string> mrcSymbols = {
      "ibv_get_device_list", "ibv_get_device_name", "ibv_open_device", "ibv_close_device", "ibv_query_qp",
      "ibv_create_cq",       "ibv_destroy_cq",      "ibv_create_qp",   "ibv_modify_qp",    "ibv_destroy_qp",
  };
#endif  // (MRC_SUPPORT)
  if (!globalIBVerbsHandle) {
    if (mscclpp::env()->ibvSo != "") {
      void* handle = ::dlopen(mscclpp::env()->ibvSo.c_str(), RTLD_NOW);
      if (handle) {
        globalIBVerbsHandle.reset(handle);
      }
    } else {
      const char* possibleLibNames[] = {"libibverbs.so", "libibverbs.so.1", nullptr};
      for (int i = 0; possibleLibNames[i] != nullptr; i++) {
        void* handle = ::dlopen(possibleLibNames[i], RTLD_NOW);
        if (handle) {
          globalIBVerbsHandle.reset(handle);
          break;
        }
      }
    }
    if (!globalIBVerbsHandle) {
      THROW(NET, SysError, errno, "Failed to open libibverbs: ", std::string(::dlerror()));
    }
  }
#if (MRC_SUPPORT)
  // In MRC mode, `VMRC_LIBIBVERBS_SO` should be set.
  char* vmrcLibibverbsSo = ::getenv("VMRC_LIBIBVERBS_SO");
  void* ptr;
  if (vmrcLibibverbsSo != nullptr && mrcSymbols.find(symbol) == mrcSymbols.end()) {
    // If we are in MRC mode and the symbol is not in the table, get it from the original libibverbs.
    if (!globalOrigIBVerbsHandle) {
      void* handle = ::dlopen(vmrcLibibverbsSo, RTLD_NOW);
      if (!handle) {
        THROW(NET, SysError, errno, "Failed to open ", std::string(vmrcLibibverbsSo));
      }
      globalOrigIBVerbsHandle.reset(handle);
    }
    ptr = ::dlsym(globalOrigIBVerbsHandle.get(), symbol.c_str());
  } else {
    ptr = ::dlsym(globalIBVerbsHandle.get(), symbol.c_str());
  }
#else   // !(MRC_SUPPORT)
  void* ptr = ::dlsym(globalIBVerbsHandle.get(), symbol.c_str());
#endif  // !(MRC_SUPPORT)
  if (!ptr && !allowReturnNull) {
    THROW(NET, SysError, errno, "Failed to load libibverbs symbol: ", symbol);
  }
  return ptr;
}

bool IBVerbs::isDmabufSupported() {
  static int isSupported = -1;
  if (isSupported == -1) {
    void* ptr = IBVerbs::dlsym("ibv_reg_dmabuf_mr", true);
    isSupported = (ptr != nullptr);
    if (!isSupported) {
      DEBUG(NET, "This platform does not support DMABUF");
    }
  }
  return isSupported;
}

struct ibv_mr* IBVerbs::ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                          int access) {
  using FuncType = struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);
  static FuncType impl = nullptr;
  if (!isDmabufSupported())
    THROW(NET, Error, ErrorCode::InvalidUsage, "libibverbs does not support ibv_reg_dmabuf_mr in this platform.");
  if (!impl) impl = reinterpret_cast<FuncType>(IBVerbs::dlsym("ibv_reg_dmabuf_mr"));
  return impl(pd, offset, length, iova, fd, access);
}

}  // namespace mscclpp
