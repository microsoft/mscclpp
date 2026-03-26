// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_MLX5DV_WRAPPER_HPP_
#define MSCCLPP_MLX5DV_WRAPPER_HPP_

#if defined(MSCCLPP_USE_MLX5DV)

#include <infiniband/mlx5dv.h>

#include <string>

namespace mscclpp {

struct MLX5DV {
  /// Whether libmlx5.so was successfully loaded at runtime.
  static bool isAvailable();

  /// Check if the given IB device supports mlx5 Direct Verbs.
  static bool mlx5dv_is_supported(struct ibv_device* device);

  /// Create a QP using mlx5dv extensions.
  static struct ibv_qp* mlx5dv_create_qp(struct ibv_context* ctx, struct ibv_qp_init_attr_ex* qpAttr,
                                          struct mlx5dv_qp_init_attr* mlx5QpAttr);

  /// Register a DMABUF memory region using mlx5dv extensions.
  /// Returns nullptr if mlx5dv_reg_dmabuf_mr is not available in this rdma-core version.
  static struct ibv_mr* mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                              int access);

 private:
  static void* dlsym(const std::string& symbol, bool allowReturnNull = false);
};

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_MLX5DV)
#endif  // MSCCLPP_MLX5DV_WRAPPER_HPP_
