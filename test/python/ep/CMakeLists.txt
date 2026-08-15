# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Standalone build for the optional in-process CUPTI kernel timer.
#
# Example:
#   cmake -S test/python/ep -B test/python/ep/build \
#     -DCMAKE_CUDA_ARCHITECTURES=90
#   cmake --build test/python/ep/build -j 64

cmake_minimum_required(VERSION 3.25)
project(mscclpp_ep_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS "${CUDAToolkit_TARGET_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}"
        "${CUDAToolkit_ROOT}/extras/CUPTI/include"
  REQUIRED)
find_library(CUPTI_LIBRARY cupti
  HINTS "${CUDAToolkit_TARGET_DIR}/lib" "${CUDAToolkit_LIBRARY_DIR}"
        "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
  REQUIRED)

add_library(cupti_kernel_timer SHARED cupti_kernel_timer.cpp)
target_include_directories(cupti_kernel_timer PRIVATE "${CUPTI_INCLUDE_DIR}")
target_link_libraries(cupti_kernel_timer PRIVATE "${CUPTI_LIBRARY}" Threads::Threads)
