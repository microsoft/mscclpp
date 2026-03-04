# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Find the MLX5 Direct Verbs (mlx5dv) library
#
# The following variables are optionally searched for defaults
#  MLX5_ROOT_DIR: Base directory where all MLX5 components are found
#  MLX5_INCLUDE_DIR: Directory where MLX5 headers are found
#  MLX5_LIB_DIR: Directory where MLX5 libraries are found

# The following are set after configuration is done:
#  MLX5_FOUND
#  MLX5_INCLUDE_DIRS
#  MLX5_LIBRARIES

find_path(MLX5_INCLUDE_DIRS
  NAMES infiniband/mlx5dv.h
  HINTS
  ${MLX5_INCLUDE_DIR}
  ${MLX5_ROOT_DIR}
  ${MLX5_ROOT_DIR}/include
  /usr/local/include
  /usr/include)

find_library(MLX5_LIBRARIES
  NAMES mlx5
  HINTS
  ${MLX5_LIB_DIR}
  ${MLX5_ROOT_DIR}
  ${MLX5_ROOT_DIR}/lib
  /usr/local/lib
  /usr/lib
  /usr/lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLX5 DEFAULT_MSG MLX5_INCLUDE_DIRS MLX5_LIBRARIES)
mark_as_advanced(MLX5_INCLUDE_DIRS MLX5_LIBRARIES)
