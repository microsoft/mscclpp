# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Builds `mscclpp_ep_cpp`, a nanobind extension that exposes the EP
# (Mixture-of-Experts dispatch/combine) runtime to Python.
#
# Two backends share one module behind a single `MoERuntime` class (`MoEMode`
# selects the path), both with a torch-free, raw-pointer (uintptr_t) nanobind
# API so the module never links libtorch:
#  - Low-latency (LL): ll_runtime.cc + low_latency/{dispatch,combine}.cu.
#  - High-throughput (HT): ht_runtime.cc + high-throughput/*.cu, a DeepEP-style
#    runtime de-torched to the same pointer boundary as the LL runtime (dynamic
#    recv sizing via the multi-step layout -> notify -> allocate -> dispatch API).

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
    ll_runtime.cc
    bindings.cpp
    low_latency/dispatch.cu
    low_latency/combine.cu
    # High-throughput (DeepEP-style) backend (torch-free, raw-pointer API).
    ht_runtime.cc
    high-throughput/counts.cu
    high-throughput/dispatch.cu
    high-throughput/combine.cu
)

# Build as a Python extension module (shared object with Python ABI suffix).
nanobind_add_module(mscclpp_ep_cpp ${EP_SOURCES})
if(MSCCLPP_USE_CUDA)
    set_target_properties(mscclpp_ep_cpp PROPERTIES CUDA_ARCHITECTURES "${_mscclpp_ep_gpu_archs}")
endif()

target_include_directories(mscclpp_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/high-throughput
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

# GPU-initiated networking (GPUNetIO / GDAKI) backend for cross-NVLink-domain LL.
# The GpuNetIoService host code is compiled into libmscclpp (mscclpp_gpunetio_obj),
# so the EP module only needs the header include path and the compile define.
# Scope both to the host runtime translation unit (ll_runtime.cc) so the EP CUDA
# device TUs are unaffected (they only forward-declare GpuNetIoDeviceContext).
if(MSCCLPP_USE_GPUNETIO)
    set_source_files_properties(ll_runtime.cc PROPERTIES
        COMPILE_DEFINITIONS MSCCLPP_USE_GPUNETIO
        INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src/gpunetio/include;${PROJECT_SOURCE_DIR}/src/gpunetio/include/host")
    # The low-latency dispatch kernel issues the inter-domain GPUNetIO sends, so it
    # needs the device context definition + the vendored DOCA device headers.
    set_source_files_properties(low_latency/dispatch.cu PROPERTIES
        COMPILE_DEFINITIONS MSCCLPP_USE_GPUNETIO
        INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src/gpunetio/include;${PROJECT_SOURCE_DIR}/src/gpunetio/include/host")
endif()
