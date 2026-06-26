# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Builds `mscclpp_ep_cpp`, a nanobind extension that exposes the EP
# (Mixture-of-Experts dispatch/combine) runtime to Python.

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
if(NOT TARGET nanobind-static AND NOT TARGET nanobind)
    FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.9.2)
    FetchContent_MakeAvailable(nanobind)
endif()

set(_mscclpp_ep_gpu_archs "")
if(MSCCLPP_USE_CUDA)
    foreach(_arch IN LISTS MSCCLPP_GPU_ARCHS)
        if(_arch STREQUAL "native")
            list(APPEND _mscclpp_ep_gpu_archs "${_arch}")
        else()
            string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
            if(_arch_num AND _arch_num GREATER_EQUAL 90)
                list(APPEND _mscclpp_ep_gpu_archs "${_arch}")
            endif()
        endif()
    endforeach()
    if(NOT _mscclpp_ep_gpu_archs)
        message(FATAL_ERROR
            "MSCCLPP_BUILD_EXT_EP=ON requires CUDA architecture 90 or newer. "
            "Set MSCCLPP_GPU_ARCHS to include 90+ (for example, -DMSCCLPP_GPU_ARCHS=90).")
    endif()
endif()

set(EP_SOURCES
    moe_runtime.cc
    bindings.cpp
    kernels/low_latency.cu
)

if(MSCCLPP_USE_ROCM)
    # ROCm port of the EP kernels is not supported yet; see src/ext/ep/README.md.
    message(WARNING "mscclpp_ep: ROCm build path not implemented, falling back to CXX compile.")
    file(GLOB_RECURSE EP_CU_SOURCES kernels/*.cu)
    set_source_files_properties(${EP_CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

# Build as a Python extension module (shared object with Python ABI suffix).
nanobind_add_module(mscclpp_ep_cpp ${EP_SOURCES})
if(MSCCLPP_USE_CUDA)
    set_target_properties(mscclpp_ep_cpp PROPERTIES CUDA_ARCHITECTURES "${_mscclpp_ep_gpu_archs}")
endif()

target_include_directories(mscclpp_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${PROJECT_SOURCE_DIR}/src/ext/include
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(mscclpp_ep_cpp PRIVATE mscclpp ${GPU_LIBRARIES} Threads::Threads)

# Number of intra-node NVLink peers compiled into the EP kernels.
#  - 8 (default) matches DeepEP upstream and H100/HGX 8-GPU nodes.
#  - 4 is required for Azure GB200 NVL72 (4 GPUs per NUMA host).
set(MSCCLPP_EP_NUM_MAX_NVL_PEERS "8" CACHE STRING
    "Compile-time NUM_MAX_NVL_PEERS for the EP kernels (8 for HGX, 4 for GB200)")
target_compile_definitions(mscclpp_ep_cpp PRIVATE
    NUM_MAX_NVL_PEERS=${MSCCLPP_EP_NUM_MAX_NVL_PEERS})

# Kernel-side debug timeout (~10s) — set via:
#   -DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT=ON
option(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT
       "Use a short ~10s kernel spin timeout (default is ~100s)" OFF)
if(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
endif()

set_target_properties(mscclpp_ep_cpp PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET default
)

# Install layout.
#  - scikit-build / wheel build (SKBUILD set by scikit-build-core):
#       module lands next to the `mscclpp` python package; libmscclpp.so is
#       under `mscclpp/lib/`, so rpath = `$ORIGIN/mscclpp/lib`.
#  - Plain CMake install: standard `${INSTALL_PREFIX}/lib` with rpath
#       `$ORIGIN/../lib` so the .so finds the sibling mscclpp shared lib.
if(DEFINED SKBUILD OR DEFINED ENV{SKBUILD})
    set_target_properties(mscclpp_ep_cpp PROPERTIES
        INSTALL_RPATH "\$ORIGIN/mscclpp/lib")
    install(TARGETS mscclpp_ep_cpp LIBRARY DESTINATION ..)
else()
    set_target_properties(mscclpp_ep_cpp PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib")
    install(TARGETS mscclpp_ep_cpp
        LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
endif()

if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_USE_ROCM)
endif()
