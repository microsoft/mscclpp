cmake_minimum_required(VERSION 3.26)
project(mscclpp LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(ENABLE_TRACE "Enable tracing" OFF)
option(USE_MPI_FOR_TESTS "Use MPI for tests" ON)
option(USE_NPKIT "Use NPKIT" ON)
option(ALLOW_GDRCOPY "Use GDRCopy, if available" OFF)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)

# Set CUDA flags based on the detected CUDA version
if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
    endif()

    message("Detected CUDA version: ${CUDAToolkit_VERSION}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

find_package(IBVerbs REQUIRED)
find_package(NUMA REQUIRED)
if(USE_MPI_FOR_TESTS)
    find_package(MPI REQUIRED)
endif()
if(ALLOW_GDRCOPY)
    find_package(GDRCopy)
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include(CTest)
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip)
option(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(CLANG_FORMAT_SOURCE_DIRS include src test)
include(${PROJECT_SOURCE_DIR}/cmake/AddClangFormatTargets.cmake)

add_library(mscclpp SHARED)
target_include_directories(mscclpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
set_target_properties(mscclpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mscclpp PRIVATE MSCCLPP::ibverbs MSCCLPP::numa CUDA::cudart CUDA::cuda_driver)
if(ENABLE_TRACE)
    target_compile_definitions(mscclpp PRIVATE ENABLE_TRACE)
endif()
if(USE_NPKIT)
    target_compile_definitions(mscclpp PRIVATE ENABLE_NPKIT)
endif()
if(ALLOW_GDRCOPY AND GDRCOPY_FOUND)
    target_compile_definitions(mscclpp PRIVATE MSCCLPP_USE_GDRCOPY)
    target_link_libraries(mscclpp PRIVATE MSCCLPP::gdrcopy)
endif()

add_subdirectory(include) # This adds the public headers to install with mscclpp
add_subdirectory(src) # This adds the sources to the mscclpp target
add_subdirectory(test)
