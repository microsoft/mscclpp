# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(MSCCLPP_MAJOR "0")
set(MSCCLPP_MINOR "6")
set(MSCCLPP_PATCH "0")

set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})
set(MSCCLPP_VERSION "${MSCCLPP_MAJOR}.${MSCCLPP_MINOR}.${MSCCLPP_PATCH}")

cmake_minimum_required(VERSION 3.25)
project(mscclpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Options
option(MSCCLPP_ENABLE_TRACE "Enable tracing" OFF)
option(MSCCLPP_BUILD_TESTS "Build tests" ON)
option(MSCCLPP_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(MSCCLPP_BUILD_APPS_NCCL "Build NCCL interfaces" ON)
option(MSCCLPP_USE_CUDA "Use NVIDIA/CUDA." OFF)
option(MSCCLPP_USE_ROCM "Use AMD/ROCm." OFF)
option(MSCCLPP_BYPASS_GPU_CHECK "Bypass GPU check." OFF)
option(MSCCLPP_NPKIT_FLAGS "Enable NPKIT" OFF)

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
    else()
        message(FATAL_ERROR "Neither NVIDIA/CUDA nor AMD/ROCm is found.")
    endif()
endif()

# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(MSCCLPP_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    enable_language(CUDA)

    # CUDA 11 or higher is required
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    # Set CUDA architectures
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    # Hopper architecture
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 90)
    endif()

    set(GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver)
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
else()
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")

    set(CMAKE_HIP_ARCHITECTURES gfx90a gfx941 gfx942)

    set(GPU_LIBRARIES hip::device)
    set(GPU_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
endif()

# Format targets
include(${PROJECT_SOURCE_DIR}/cmake/AddFormatTargets.cmake)

# Find ibverbs and libnuma
find_package(IBVerbs)
find_package(NUMA REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_library(mscclpp_obj OBJECT)
target_include_directories(mscclpp_obj
    SYSTEM PRIVATE
    ${GPU_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS})
target_link_libraries(mscclpp_obj PRIVATE ${GPU_LIBRARIES} ${NUMA_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads dl)
if(IBVERBS_FOUND)
    target_include_directories(mscclpp_obj SYSTEM PRIVATE ${IBVERBS_INCLUDE_DIRS})
    target_link_libraries(mscclpp_obj PRIVATE ${IBVERBS_LIBRARIES})
    target_compile_definitions(mscclpp_obj PUBLIC USE_IBVERBS)
endif()
set_target_properties(mscclpp_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_ROCM)
endif()
if(MSCCLPP_ENABLE_TRACE)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_ENABLE_TRACE)
endif()
if(MSCCLPP_NPKIT_FLAGS)
    target_compile_definitions(mscclpp_obj PRIVATE ${MSCCLPP_NPKIT_FLAGS})
endif()

# libmscclpp
add_library(mscclpp SHARED)
target_link_libraries(mscclpp PUBLIC mscclpp_obj)
set_target_properties(mscclpp PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclpp_static STATIC)
target_link_libraries(mscclpp_static PUBLIC mscclpp_obj)
set_target_properties(mscclpp_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_subdirectory(include)
add_subdirectory(src)

if("${INSTALL_PREFIX}" STREQUAL "")
    set(INSTALL_PREFIX "./")
endif()

install(TARGETS mscclpp_obj
    FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
install(TARGETS mscclpp
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_static
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

# Tests
if(MSCCLPP_BUILD_TESTS)
    enable_testing() # Called here to allow ctest from the build directory
    add_subdirectory(test)
endif()

# Python bindings
if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# NCCL interfaces
if(MSCCLPP_BUILD_APPS_NCCL)
    add_subdirectory(apps/nccl)
endif()
