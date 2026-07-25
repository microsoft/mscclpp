# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Standalone build for:
#   - mscclpp_ep_bench: pure-C++/MPI low-latency EP benchmark.
#   - cupti_kernel_timer: optional in-process timer used by the LL benchmarks.
#
# Example:
#   cmake -S test/python/ep -B test/python/ep/build \
#     -DCMAKE_CUDA_ARCHITECTURES=90
#   cmake --build test/python/ep/build -j 64

cmake_minimum_required(VERSION 3.25)
project(mscclpp_ep_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

get_filename_component(_default_mscclpp_src "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(MSCCLPP_SRC "${_default_mscclpp_src}" CACHE PATH "mscclpp source tree")

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

if(NOT DEFINED MSCCLPP_INSTALL_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import os, mscclpp._mscclpp as m; print(os.path.dirname(m.__file__))"
    OUTPUT_VARIABLE MSCCLPP_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _mscclpp_python_result)
  if(NOT _mscclpp_python_result EQUAL 0)
    unset(MSCCLPP_INSTALL_DIR)
  endif()
endif()
if(NOT MSCCLPP_INSTALL_DIR)
  message(FATAL_ERROR
          "MSCCLPP_INSTALL_DIR not found; install mscclpp or pass the package directory containing lib/ and include/")
endif()

set(EP "${MSCCLPP_SRC}/src/ext/ep")
find_library(MSCCLPP_LIBRARY NAMES mscclpp HINTS "${MSCCLPP_INSTALL_DIR}/lib" REQUIRED)
find_path(CUPTI_INCLUDE_DIR cupti.h
  HINTS "${CUDAToolkit_TARGET_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}"
        "${CUDAToolkit_ROOT}/extras/CUPTI/include"
  REQUIRED)
find_library(CUPTI_LIBRARY cupti
  HINTS "${CUDAToolkit_TARGET_DIR}/lib" "${CUDAToolkit_LIBRARY_DIR}"
        "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
  REQUIRED)

add_executable(mscclpp_ep_bench
  mscclpp_ep_bench.cu
  "${EP}/moe_runtime.cc"
  "${EP}/low_latency/dispatch.cu"
  "${EP}/low_latency/combine.cu")

target_include_directories(mscclpp_ep_bench PRIVATE
  "${EP}"
  "${EP}/include"
  "${MSCCLPP_SRC}/include"
  "${MSCCLPP_SRC}/src/core/include"
  "${MSCCLPP_SRC}/src/ext/include"
  "${MSCCLPP_INSTALL_DIR}/include"
  "${CUPTI_INCLUDE_DIR}")

target_compile_definitions(mscclpp_ep_bench PRIVATE MSCCLPP_USE_CUDA)

target_compile_options(mscclpp_ep_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

set_target_properties(mscclpp_ep_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${MSCCLPP_INSTALL_DIR}/lib"
  INSTALL_RPATH "${MSCCLPP_INSTALL_DIR}/lib")

target_link_libraries(mscclpp_ep_bench PRIVATE
  "${MSCCLPP_LIBRARY}"
  "${CUPTI_LIBRARY}"
  MPI::MPI_CXX
  CUDA::cudart
  CUDA::cuda_driver
  Threads::Threads)

add_library(cupti_kernel_timer SHARED cupti_kernel_timer.cpp)
target_include_directories(cupti_kernel_timer PRIVATE "${CUPTI_INCLUDE_DIR}")
target_link_libraries(cupti_kernel_timer PRIVATE "${CUPTI_LIBRARY}" Threads::Threads)
