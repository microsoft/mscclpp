cmake_minimum_required(VERSION 3.26)

project(mscclpp LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(CUDAToolkit REQUIRED)
find_package(IBVerbs REQUIRED)
find_package(NUMA REQUIRED)
find_package(GDRCopy)

option(USE_MPI_FOR_TESTS "Use MPI for tests" ON)
if(USE_MPI_FOR_TESTS)
    find_package(MPI REQUIRED)
    add_definitions(-DMSCCLPP_USE_MPI_FOR_TESTS)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(mscclpp SHARED)
target_include_directories(mscclpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
set_target_properties(mscclpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mscclpp PRIVATE MSCCLPP::ibverbs MSCCLPP::numa CUDA::cudart CUDA::cuda_driver)
if(GDRCOPY_FOUND)
    target_link_libraries(mscclpp PRIVATE MSCCLPP::gdrcopy)
endif()

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    message(STATUS "Found clang-format: ${CLANG_FORMAT}")
    file(GLOB_RECURSE ALL_SOURCES *.h *.hpp *.cc *.cpp *.cu)
    add_custom_target(check-format ALL COMMAND ${CLANG_FORMAT} --dry-run ${ALL_SOURCES})
    add_custom_target(format COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCES})
else()
    message(STATUS "clang-format not found.")
endif()

add_subdirectory(src) # This adds the sources to the mscclpp target
add_subdirectory(tests)
