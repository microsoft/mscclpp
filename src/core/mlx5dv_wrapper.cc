// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#if defined(MSCCLPP_USE_MLX5DV)

// _GNU_SOURCE is required for dlvsym()
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mlx5dv_wrapper.hpp"

#include <dlfcn.h>
#include <infiniband/mlx5dv.h>

#ifndef MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT
#define MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT (1 << 0)
#endif

#include <memory>

#include "logger.hpp"

namespace mscclpp {

static std::unique_ptr<void, int (*)(void*)> globalMLX5Handle(nullptr, &::dlclose);

void* MLX5DV::dlsym(const std::string& symbol, bool allowReturnNull) {
  if (!globalMLX5Handle) {
    const char* possibleLibNames[] = {"libmlx5.so", "libmlx5.so.1", nullptr};
    for (int i = 0; possibleLibNames[i] != nullptr; i++) {
      void* handle = ::dlopen(possibleLibNames[i], RTLD_NOW);
      if (handle) {
        globalMLX5Handle.reset(handle);
        break;
      }
    }
    if (!globalMLX5Handle) {
      if (allowReturnNull) return nullptr;
      THROW(NET, SysError, errno, "Failed to open libmlx5: ", std::string(::dlerror()));
    }
  }
  void* ptr = ::dlsym(globalMLX5Handle.get(), symbol.c_str());
  if (!ptr && !allowReturnNull) {
    THROW(NET, SysError, errno, "Failed to load libmlx5 symbol: ", symbol);
  }
  return ptr;
}

bool MLX5DV::isAvailable() {
  static int available = -1;
  if (available == -1) {
    // Try to load the library; if it fails, mlx5dv is not available
    const char* possibleLibNames[] = {"libmlx5.so", "libmlx5.so.1", nullptr};
    for (int i = 0; possibleLibNames[i] != nullptr; i++) {
      void* handle = ::dlopen(possibleLibNames[i], RTLD_NOW);
      if (handle) {
        if (!globalMLX5Handle) {
          globalMLX5Handle.reset(handle);
        } else {
          ::dlclose(handle);
        }
        available = 1;
        INFO(NET, "libmlx5 loaded successfully");
        return true;
      }
    }
    available = 0;
    DEBUG(NET, "libmlx5 not available");
  }
  return available == 1;
}

bool MLX5DV::mlx5dv_is_supported(struct ibv_device* device) {
  using FuncType = bool (*)(struct ibv_device*);
  static FuncType impl = nullptr;
  if (!impl) {
    void* ptr = MLX5DV::dlsym("mlx5dv_is_supported", /*allowReturnNull=*/true);
    if (!ptr) return false;
    impl = reinterpret_cast<FuncType>(ptr);
  }
  return impl(device);
}

struct ibv_mr* MLX5DV::mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                            int access) {
  // mlx5dv_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access) — the last arg is mlx5-specific flags.
  // Must use dlvsym with "MLX5_1.25" version to get the Data Direct-capable symbol.
  using FuncType = struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int);
  static FuncType impl = nullptr;
  static bool resolved = false;
  if (!resolved) {
    if (globalMLX5Handle) {
      void* ptr = dlvsym(globalMLX5Handle.get(), "mlx5dv_reg_dmabuf_mr", "MLX5_1.25");
      if (!ptr) {
        ptr = MLX5DV::dlsym("mlx5dv_reg_dmabuf_mr", /*allowReturnNull=*/true);
      }
      impl = ptr ? reinterpret_cast<FuncType>(ptr) : nullptr;
    }
    resolved = true;
  }
  if (!impl) return nullptr;
  return impl(pd, offset, length, iova, fd, access, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
}

int MLX5DV::mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len) {
  using FuncType = int (*)(struct ibv_context*, char*, size_t);
  static FuncType impl = nullptr;
  static bool resolved = false;
  if (!resolved) {
    if (globalMLX5Handle) {
      void* ptr = dlvsym(globalMLX5Handle.get(), "mlx5dv_get_data_direct_sysfs_path", "MLX5_1.25");
      if (!ptr) {
        ptr = MLX5DV::dlsym("mlx5dv_get_data_direct_sysfs_path", /*allowReturnNull=*/true);
      }
      impl = ptr ? reinterpret_cast<FuncType>(ptr) : nullptr;
    }
    resolved = true;
  }
  if (!impl) return -1;
  return impl(context, buf, buf_len);
}

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_MLX5DV)
