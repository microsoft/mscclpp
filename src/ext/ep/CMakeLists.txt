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
