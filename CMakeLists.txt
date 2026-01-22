# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.25)
project(mscclpp LANGUAGES CXX)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" MSCCLPP_VERSION_CONTENT)
if(MSCCLPP_VERSION_CONTENT MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(MSCCLPP_MAJOR "${CMAKE_MATCH_1}")
    set(MSCCLPP_MINOR "${CMAKE_MATCH_2}")
    set(MSCCLPP_PATCH "${CMAKE_MATCH_3}")
else()
    message(FATAL_ERROR "VERSION file must be in the format MAJOR.MINOR.PATCH")
endif()

set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})
set(MSCCLPP_VERSION "${MSCCLPP_MAJOR}.${MSCCLPP_MINOR}.${MSCCLPP_PATCH}")

# Set output directories for all targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Git)
set(GIT_HASH "UNKNOWN")
if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _git_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _git_out STREQUAL "")
        set(GIT_HASH "${_git_out}")
    endif()
else()
    message(WARNING "Git not found, setting GIT_HASH to 'UNKNOWN'")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/mscclpp/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/mscclpp/version.hpp"
    @ONLY
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Options
option(MSCCLPP_ENABLE_TRACE "Enable tracing" OFF)
option(MSCCLPP_BUILD_TESTS "Build tests" ON)
option(MSCCLPP_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(MSCCLPP_BUILD_EXT_NCCL "Build NCCL interfaces" ON)
option(MSCCLPP_BUILD_EXT_COLLECTIVES "Build collective algorithms" ON)
option(MSCCLPP_USE_CUDA "Use NVIDIA/CUDA." OFF)
option(MSCCLPP_USE_ROCM "Use AMD/ROCm." OFF)
option(MSCCLPP_USE_IB "Use InfiniBand." ON)
option(MSCCLPP_BYPASS_GPU_CHECK "Bypass GPU check." OFF)
option(MSCCLPP_NPKIT_FLAGS "Set NPKIT flags" OFF)
set(MSCCLPP_GPU_ARCHS "" CACHE STRING "Specify GPU architectures with delimiters (comma, space, or semicolon).")

if(MSCCLPP_BYPASS_GPU_CHECK)
    if(MSCCLPP_USE_CUDA)
        message(STATUS "Bypassing GPU check: using NVIDIA/CUDA.")
        find_package(CUDAToolkit REQUIRED)
    elseif(MSCCLPP_USE_ROCM)
        message(STATUS "Bypassing GPU check: using AMD/ROCm.")
        # Temporal fix for rocm5.6
        set(CMAKE_PREFIX_PATH "/opt/rocm;${CMAKE_PREFIX_PATH}")
        find_package(hip REQUIRED)
    else()
        message(FATAL_ERROR "Bypassing GPU check: neither NVIDIA/CUDA nor AMD/ROCm is specified.")
    endif()
else()
    # Detect GPUs
    include(CheckNvidiaGpu)
    include(CheckAmdGpu)
    if(NVIDIA_FOUND AND AMD_FOUND)
        message(STATUS "Detected NVIDIA/CUDA and AMD/ROCm: prioritizing NVIDIA/CUDA.")
        set(MSCCLPP_USE_CUDA ON)
        set(MSCCLPP_USE_ROCM OFF)
    elseif(NVIDIA_FOUND)
        message(STATUS "Detected NVIDIA/CUDA.")
        set(MSCCLPP_USE_CUDA ON)
        set(MSCCLPP_USE_ROCM OFF)
    elseif(AMD_FOUND)
        message(STATUS "Detected AMD/ROCm.")
        set(MSCCLPP_USE_CUDA OFF)
        set(MSCCLPP_USE_ROCM ON)
    elseif(CUDAToolkit_FOUND)
        message(WARNING "CUDAToolkit found but no compatible GPU detected. Defaulting to CUDA.")
        set(MSCCLPP_USE_CUDA ON)
        set(MSCCLPP_USE_ROCM OFF)
    elseif(hip_FOUND)
        message(WARNING "HIP found but no compatible GPU detected. Defaulting to ROCm.")
        set(MSCCLPP_USE_CUDA OFF)
        set(MSCCLPP_USE_ROCM ON)
    else()
        message(FATAL_ERROR "No compatible GPU found. Set MSCCLPP_USE_CUDA or MSCCLPP_USE_ROCM to ON.")
    endif()
endif()

#Function to choose between default and "a" variant.
function(mscclpp_choose_arch_variant out_arch base_arch a_arch)
  # base_arch like "90"  a_arch like "90a"
  set(PROBE_SRC "${CMAKE_SOURCE_DIR}/cmake/arch_probe.cu")

  # Unique object name per arch to avoid collisions
  set(PROBE_OBJ "${CMAKE_BINARY_DIR}/CMakeFiles/arch_probe_sm${a_arch}.o")

  # Force explicit codegen so -arch=native can't override.
  # Example for 90a: compute_90a/sm_90a
  set(GENCODE_A "--generate-code=arch=compute_${a_arch},code=[compute_${a_arch},sm_${a_arch}]")

  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}"
      -c "${PROBE_SRC}"
      -o "${PROBE_OBJ}"
      "${GENCODE_A}"
      -O3 -DNDEBUG
      -Xcompiler=-fPIC
    RESULT_VARIABLE PROBE_RC
    OUTPUT_QUIET
    ERROR_VARIABLE PROBE_STDERR
  )

  if(PROBE_RC EQUAL 0)
    message(STATUS "Arch probe OK for sm_${a_arch} -> selecting ${a_arch}")
    set(${out_arch} "${a_arch}" PARENT_SCOPE)
  else()
    message(STATUS "Arch probe FAILED for sm_${a_arch} -> selecting ${base_arch}")
    # Uncomment if you want details:
    # message(STATUS "Probe stderr:\n${PROBE_STDERR}")
    set(${out_arch} "${base_arch}" PARENT_SCOPE)
  endif()
endfunction()

if(MSCCLPP_GPU_ARCHS)
    string(STRIP "${MSCCLPP_GPU_ARCHS}" MSCCLPP_GPU_ARCHS)
    string(REPLACE " " ";" MSCCLPP_GPU_ARCHS "${MSCCLPP_GPU_ARCHS}")
    string(REPLACE "," ";" MSCCLPP_GPU_ARCHS "${MSCCLPP_GPU_ARCHS}")
    if(NOT MSCCLPP_GPU_ARCHS)
        message(FATAL_ERROR "MSCCLPP_GPU_ARCHS is empty. Specify GPU architectures or leave unset.")
    endif()
elseif(MSCCLPP_USE_CUDA)
    if(CUDAToolkit_VERSION VERSION_LESS "11.8")
        message(FATAL_ERROR "CUDA 11.8 or higher required, found ${CUDAToolkit_VERSION}")
    endif()
    set(MSCCLPP_GPU_ARCHS 80)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
	mscclpp_choose_arch_variant(MSCCLPP_SM100_VARIANT "90" "90a")
	list(APPEND MSCCLPP_GPU_ARCHS ${MSCCLPP_SM90_VARIANT})
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
	mscclpp_choose_arch_variant(MSCCLPP_SM100_VARIANT "100" "100a")
	list(APPEND MSCCLPP_GPU_ARCHS ${MSCCLPP_SM100_VARIANT})
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
        mscclpp_choose_arch_variant(MSCCLPP_SM100_VARIANT "120" "120a")
	list(APPEND MSCCLPP_GPU_ARCHS ${MSCCLPP_SM120_VARIANT})
    endif()
elseif(MSCCLPP_USE_ROCM)
    set(MSCCLPP_GPU_ARCHS gfx90a gfx941 gfx942)
endif()

message(STATUS "GPU architectures: ${MSCCLPP_GPU_ARCHS}")

# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(MSCCLPP_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    enable_language(CUDA)

    set(CMAKE_CUDA_ARCHITECTURES ${MSCCLPP_GPU_ARCHS})

    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})

    find_package(CCCL PATHS ${CUDAToolkit_LIBRARY_DIR}/cmake/cccl)
    if (CCCL_FOUND AND CUDAToolkit_VERSION_MAJOR GREATER 12)
        set(GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver CCCL::CCCL)
    else()
        set(GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver)
    endif()
else()
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")

    set(CMAKE_HIP_ARCHITECTURES ${MSCCLPP_GPU_ARCHS})

    set(GPU_LIBRARIES hip::device)
    set(GPU_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_BUILD)
endif()

if(MSCCLPP_USE_IB)
    find_package(IBVerbs)
    if(NOT IBVERBS_FOUND)
        message(FATAL_ERROR "IBVerbs not found. Install libibverbs-dev or rdma-core-devel. If you want to disable InfiniBand, add `-DMSCCLPP_USE_IB=OFF` in your cmake command.")
    endif()
endif()
find_package(NUMA REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

if("${INSTALL_PREFIX}" STREQUAL "")
    set(INSTALL_PREFIX "./")
endif()

add_subdirectory(src)
add_subdirectory(include)

# Tests
if(MSCCLPP_BUILD_TESTS)
    enable_testing() # Called here to allow ctest from the build directory
    add_subdirectory(test)
endif()

# Python bindings
if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
