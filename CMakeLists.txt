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
add_subdirectory(src) # This adds the srouces to the mscclpp target
target_include_directories(mscclpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
set_target_properties(mscclpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mscclpp PRIVATE MSCCLPP::ibverbs MSCCLPP::numa CUDA::cudart CUDA::cuda_driver)
if(GDRCOPY_FOUND)
    target_link_libraries(mscclpp PRIVATE MSCCLPP::gdrcopy)
endif()

add_subdirectory(tests)
