# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(_mscclpp_ep_gpu_archs "")
get_property(_mscclpp_requested_gpu_archs CACHE MSCCLPP_GPU_ARCHS PROPERTY VALUE)
if(_mscclpp_requested_gpu_archs)
    foreach(_arch IN LISTS MSCCLPP_GPU_ARCHS)
        if(_arch STREQUAL "native")
            list(APPEND _mscclpp_ep_gpu_archs native)
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

set(MSCCLPP_EP_SOURCES
    combine/direct_send_combine.cu
    combine/rank_local_reduce_combine.cu
    dispatch/expert_major_dispatch.cu
    dispatch/rank_major_dispatch.cu
    latency.cc
    moe_runtime.cc
)

add_library(mscclpp_ep SHARED ${MSCCLPP_EP_SOURCES})

target_include_directories(mscclpp_ep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(mscclpp_ep PUBLIC mscclpp)
target_link_libraries(mscclpp_ep PRIVATE ${GPU_LIBRARIES} Threads::Threads)
target_compile_definitions(mscclpp_ep PRIVATE MSCCLPP_USE_CUDA)

set_target_properties(mscclpp_ep PROPERTIES
    CUDA_ARCHITECTURES "${_mscclpp_ep_gpu_archs}"
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    VERSION ${MSCCLPP_VERSION}
    SOVERSION ${MSCCLPP_SOVERSION}
)

install(TARGETS mscclpp_ep
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
