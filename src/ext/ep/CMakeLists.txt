# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Builds `mscclpp_ep_cpp`, a nanobind extension that exposes the EP
# (Mixture-of-Experts dispatch/combine) runtime to Python.
#
# One `MoERuntime` conditionally owns the resources required by the selected
# latency or overlap algorithms. All kernels consume the same persistent
# `DeviceContext*` through a torch-free raw-pointer nanobind API.

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
if(NOT TARGET nanobind-static AND NOT TARGET nanobind)
    FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.9.2)
    FetchContent_MakeAvailable(nanobind)
endif()

set(_mscclpp_ep_gpu_archs "")
if(MSCCLPP_USE_CUDA)
    get_property(_mscclpp_requested_gpu_archs CACHE MSCCLPP_GPU_ARCHS PROPERTY VALUE)
    if(_mscclpp_requested_gpu_archs)
        foreach(_arch IN LISTS MSCCLPP_GPU_ARCHS)
            if(_arch STREQUAL "native")
                foreach(_native_arch IN LISTS NVIDIA_GPU_ARCHS)
                    if(_native_arch GREATER_EQUAL 90)
                        list(APPEND _mscclpp_ep_gpu_archs "${_native_arch}")
                    endif()
                endforeach()
            else()
                string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
                if(_arch_num AND _arch_num GREATER_EQUAL 90)
                    list(APPEND _mscclpp_ep_gpu_archs "${_arch}")
                endif()
            endif()
        endforeach()
    else()
        list(APPEND _mscclpp_ep_gpu_archs 90)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND _mscclpp_ep_gpu_archs 100 100a)
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
            list(APPEND _mscclpp_ep_gpu_archs 103 103a)
        endif()
    endif()
    if(NOT _mscclpp_ep_gpu_archs)
        message(STATUS
            "Skipping the EP extension because MSCCLPP_GPU_ARCHS contains no "
            "explicit CUDA architecture 90 or newer.")
        return()
    endif()
    list(REMOVE_DUPLICATES _mscclpp_ep_gpu_archs)
    message(STATUS "EP CUDA architectures: ${_mscclpp_ep_gpu_archs}")
endif()

set(EP_SOURCES
    moe_runtime.cc
    runtime/fixed_buffer.cc
    runtime/recv_pool.cc
    bindings.cpp
    dispatch/latency/expert_major.cu
    dispatch/latency/rank_major.cu
    combine/latency/rank_local_reduce.cu
    combine/latency/direct_send.cu
    dispatch/overlap/token_major_prepare.cu
    dispatch/overlap/token_major.cu
    combine/overlap/token_major_reduce.cu
)

# Build as a Python extension module (shared object with Python ABI suffix).
nanobind_add_module(mscclpp_ep_cpp ${EP_SOURCES})
if(MSCCLPP_USE_CUDA)
    set_target_properties(mscclpp_ep_cpp PROPERTIES CUDA_ARCHITECTURES "${_mscclpp_ep_gpu_archs}")
endif()

target_include_directories(mscclpp_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${PROJECT_SOURCE_DIR}/src/ext/include
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(mscclpp_ep_cpp PRIVATE mscclpp ${GPU_LIBRARIES} Threads::Threads)

# The EP CUDA kernels call constexpr __host__ helpers (std::max / std::pair) and
# use extended lambdas in device code. These flags were previously injected
# implicitly by Torch's CMake package; add them explicitly now that the module
# no longer links libtorch.
if(MSCCLPP_USE_CUDA)
    target_compile_options(mscclpp_ep_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
endif()

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
