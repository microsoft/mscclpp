// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_MLX5DV_WRAPPER_HPP_
#define MSCCLPP_MLX5DV_WRAPPER_HPP_

#if defined(MSCCLPP_USE_MLX5DV)

#include <infiniband/verbs.h>

#include <string>

namespace mscclpp {

struct MLX5DV {
  /// Whether libmlx5.so was successfully loaded at runtime.
  static bool isAvailable();

  /// Check if the given IB device supports mlx5 Direct Verbs.
  static bool mlx5dv_is_supported(struct ibv_device* device);

  /// Register a DMABUF memory region using mlx5dv extensions.
  /// Returns nullptr if mlx5dv_reg_dmabuf_mr is not available in this rdma-core version.
  static struct ibv_mr* mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                             int access);

  /// Query the Data Direct sysfs path for the given IB context.
  /// Returns 0 on success (device supports Data Direct), non-zero otherwise.
  static int mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len);

 private:
  static void* dlsym(const std::string& symbol, bool allowReturnNull = false);
};

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_MLX5DV)
#endif  // MSCCLPP_MLX5DV_WRAPPER_HPP_
