# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(MSCCLPP_MAJOR "0")
set(MSCCLPP_MINOR "2")
set(MSCCLPP_PATCH "0")

set(MSCCLPP_SOVERSION ${MSCCLPP_MAJOR})
set(MSCCLPP_VERSION "${MSCCLPP_MAJOR}.${MSCCLPP_MINOR}.${MSCCLPP_PATCH}")

cmake_minimum_required(VERSION 3.26)
project(mscclpp LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# clang-format targets
include(${PROJECT_SOURCE_DIR}/cmake/AddClangFormatTargets.cmake)

# Options
option(ENABLE_TRACE "Enable tracing" OFF)
option(USE_NPKIT "Use NPKIT" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(ALLOW_GDRCOPY "Use GDRCopy, if available" OFF)

# Find CUDAToolkit. Set CUDA flags based on the detected CUDA version
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 90)
    endif()
endif()
set(CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver)

# Find ibverbs and libnuma
find_package(IBVerbs REQUIRED)
find_package(NUMA REQUIRED)

# Find optional packages
if(ALLOW_GDRCOPY)
    find_package(GDRCopy)
endif()

add_library(mscclppObj OBJECT)
target_include_directories(mscclppObj
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${IBVERBS_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS}
    ${GDRCOPY_INCLUDE_DIRS})
target_link_libraries(mscclppObj PRIVATE ${CUDA_LIBRARIES} ${NUMA_LIBRARIES} ${IBVERBS_LIBRARIES} ${GDRCOPY_LIBRARIES})
set_target_properties(mscclppObj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
if(ENABLE_TRACE)
    target_compile_definitions(mscclppObj PRIVATE ENABLE_TRACE)
endif()
if(USE_NPKIT)
    target_compile_definitions(mscclppObj PRIVATE ENABLE_NPKIT)
endif()
if(ALLOW_GDRCOPY AND GDRCOPY_FOUND)
    target_compile_definitions(mscclppObj PRIVATE MSCCLPP_USE_GDRCOPY)
    target_link_libraries(mscclppObj PRIVATE MSCCLPP::gdrcopy)
endif()

# libmscclpp
add_library(mscclpp SHARED)
target_link_libraries(mscclpp PUBLIC mscclppObj)
set_target_properties(mscclpp PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclppStatic STATIC)
target_link_libraries(mscclppStatic PUBLIC mscclppObj)
set_target_properties(mscclppStatic PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_subdirectory(include)
add_subdirectory(src)
install(TARGETS mscclpp mscclppStatic
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Tests
if (BUILD_TESTS)
    add_subdirectory(test)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.4.0)
    FetchContent_MakeAvailable(nanobind)
    add_subdirectory(python)
endif()
