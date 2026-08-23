# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
if(NOT TARGET nanobind-static AND NOT TARGET nanobind)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v1.9.2
    )
    FetchContent_MakeAvailable(nanobind)
endif()

set(_mscclpp_ep_gpu_archs "")
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

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cu)

nanobind_add_module(mscclpp_ep_cpp ${SOURCES})

target_include_directories(mscclpp_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(mscclpp_ep_cpp PRIVATE mscclpp ${GPU_LIBRARIES} Threads::Threads)
target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_USE_CUDA)

set_target_properties(mscclpp_ep_cpp PROPERTIES
    CUDA_ARCHITECTURES "${_mscclpp_ep_gpu_archs}"
    INSTALL_RPATH "\$ORIGIN/lib"
)

install(TARGETS mscclpp_ep_cpp LIBRARY DESTINATION .)

# Kernel-side debug timeout (~10s) — set via:
#   -DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT=ON
option(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT
       "Use a short ~10s kernel spin timeout (default is ~100s)" OFF)
if(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
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
    # The low-latency combine kernel RDMA-reads (gin->get) remote expert outputs
    # for cross-domain peers.
    set_source_files_properties(low_latency/combine.cu PROPERTIES
        COMPILE_DEFINITIONS MSCCLPP_USE_GPUNETIO
        INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src/gpunetio/include;${PROJECT_SOURCE_DIR}/src/gpunetio/include/host")
endif()
