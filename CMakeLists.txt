# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.25)

# Read version from file
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" MSCCLPP_VERSION)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${MSCCLPP_VERSION})
set(MSCCLPP_MAJOR ${CMAKE_MATCH_1})
set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})

project(mscclpp VERSION ${MSCCLPP_VERSION} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Build options
option(MSCCLPP_ENABLE_TRACE "Enable tracing" OFF)
option(MSCCLPP_BUILD_TESTS "Build tests" ON)
option(MSCCLPP_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(MSCCLPP_BUILD_APPS_NCCL "Build NCCL interfaces" ON)
option(MSCCLPP_USE_CUDA "Use NVIDIA/CUDA" OFF)
option(MSCCLPP_USE_ROCM "Use AMD/ROCm" OFF)
option(MSCCLPP_BYPASS_GPU_CHECK "Bypass GPU check" OFF)
option(MSCCLPP_NPKIT_FLAGS "Set NPKIT flags" OFF)
option(MSCCLPP_SUPPRESS_EXTERNAL_WARNINGS "Suppress warnings from external dependencies" ON)
set(MSCCLPP_GPU_ARCHS "" CACHE STRING "GPU architectures (comma/space/semicolon separated)")

# Set compiler flags
include(CompilerFlags)

# GPU detection and configuration
include(ConfigureGpu)

# Format targets
include(AddFormatTargets)

# External dependencies
include(ExternalDependencies)

# Core library configuration
include(LibraryTargets)

# Add source directories
add_subdirectory(include)
add_subdirectory(src)

# Installation
set(INSTALL_PREFIX "./" CACHE STRING "Installation prefix")
install(TARGETS mscclpp_obj FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
install(TARGETS mscclpp LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_static ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

# Optional components
if(MSCCLPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(MSCCLPP_BUILD_APPS_NCCL)
    add_subdirectory(apps/nccl)
endif()
