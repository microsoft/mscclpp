# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Standalone CMake build for mscclpp_ep_bench -- the pure-C++/MPI low-latency EP
# benchmark that calls mscclpp::ep::MoERuntime directly.
#
# The EP dispatch/combine symbols live only in the nanobind Python module
# (mscclpp_ep_cpp.so), so this recompiles the two LL translation units
# (moe_runtime.cc + kernels/low_latency.cu) into the binary and links the
# installed libmscclpp.so. Flags mirror src/ext/ep/CMakeLists.txt.
#
# Configure/build (from this directory), e.g. on a GB200 node in the torch env:
#   cmake -S . -B build \
#     -DMSCCLPP_SRC=/opt/microsoft/mrc/ep/mscclpp \
#     -DMSCCLPP_EP_NUM_MAX_NVL_PEERS=4 -DCMAKE_CUDA_ARCHITECTURES=100
#   cmake --build build -j
#
# Key cache options:
#   MSCCLPP_SRC                  mscclpp source tree (for EP headers + the two LL TUs)
#   MSCCLPP_INSTALL_DIR          installed mscclpp package dir (has lib/ + include/);
#                                autodetected under the active conda env if unset
#   MSCCLPP_EP_NUM_MAX_NVL_PEERS 4 for GB200 NVL72, 8 for HGX (default 4)
#   CMAKE_CUDA_ARCHITECTURES     100 for GB200 sm_100 (default 100)

cmake_minimum_required(VERSION 3.25)
project(mscclpp_ep_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 100)  # GB200 sm_100
endif()

# --- Paths -----------------------------------------------------------------
set(MSCCLPP_SRC "/opt/microsoft/mrc/ep/mscclpp" CACHE PATH "mscclpp source tree")
set(MSCCLPP_EP_NUM_MAX_NVL_PEERS "4" CACHE STRING
    "Compile-time NUM_MAX_NVL_PEERS for the EP kernels (4 for GB200, 8 for HGX)")

# Installed mscclpp (libmscclpp.so + public headers). Default: active conda env.
if(NOT DEFINED MSCCLPP_INSTALL_DIR)
  set(_conda "$ENV{CONDA_PREFIX}")
  if(NOT _conda)
    set(_conda "$ENV{HOME}/miniconda3/envs/torch")
  endif()
  file(GLOB _cands "${_conda}/lib/python*/site-packages/mscclpp")
  if(_cands)
    list(GET _cands 0 MSCCLPP_INSTALL_DIR)
  endif()
endif()
if(NOT MSCCLPP_INSTALL_DIR)
  message(FATAL_ERROR "MSCCLPP_INSTALL_DIR not found; pass -DMSCCLPP_INSTALL_DIR=<...>/site-packages/mscclpp")
endif()
message(STATUS "mscclpp source : ${MSCCLPP_SRC}")
message(STATUS "mscclpp install: ${MSCCLPP_INSTALL_DIR}")

set(EP "${MSCCLPP_SRC}/src/ext/ep")

find_library(MSCCLPP_LIBRARY NAMES mscclpp HINTS "${MSCCLPP_INSTALL_DIR}/lib" REQUIRED)

# --- Dependencies ----------------------------------------------------------
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUPTI (kernel timing). Not a CUDAToolkit:: imported target on all versions.
find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS "${CUDAToolkit_TARGET_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}"
        /usr/local/cuda/targets/sbsa-linux/include /usr/local/cuda/extras/CUPTI/include
  REQUIRED)
find_library(CUPTI_LIBRARY cupti
  HINTS "${CUDAToolkit_LIBRARY_DIR}" /usr/local/cuda/targets/sbsa-linux/lib
        /usr/local/cuda/extras/CUPTI/lib64
  REQUIRED)

# --- Target ----------------------------------------------------------------
add_executable(mscclpp_ep_bench
  mscclpp_ep_bench.cu
  ${EP}/moe_runtime.cc
  ${EP}/kernels/low_latency.cu
)

# moe_runtime.cc uses device intrinsics via gpu_data_types.hpp -> it is a CUDA
# translation unit (the nanobind build compiles it with nvcc too).
set_source_files_properties(${EP}/moe_runtime.cc PROPERTIES LANGUAGE CUDA)

target_include_directories(mscclpp_ep_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EP}
  ${EP}/ht
  ${MSCCLPP_SRC}/include
  ${MSCCLPP_SRC}/src/core/include
  ${MSCCLPP_SRC}/src/ext/include
  ${MSCCLPP_INSTALL_DIR}/include
  ${CUPTI_INCLUDE_DIR}
)

target_compile_definitions(mscclpp_ep_bench PRIVATE
  MSCCLPP_USE_CUDA
  EP_DISPATCH_NCCLEP
  NUM_MAX_NVL_PEERS=${MSCCLPP_EP_NUM_MAX_NVL_PEERS}
)

target_compile_options(mscclpp_ep_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

set_target_properties(mscclpp_ep_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${MSCCLPP_INSTALL_DIR}/lib"
  INSTALL_RPATH "${MSCCLPP_INSTALL_DIR}/lib"
)

target_link_libraries(mscclpp_ep_bench PRIVATE
  ${MSCCLPP_LIBRARY}
  ${CUPTI_LIBRARY}
  MPI::MPI_CXX
  CUDA::cudart
  CUDA::cuda_driver
)
