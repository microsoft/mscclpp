# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(MPI)

set(TEST_LIBS_COMMON mscclpp ${GPU_LIBRARIES} ${NUMA_LIBRARIES} Threads::Threads)
if(MSCCLPP_USE_IB)
    list(APPEND TEST_LIBS_COMMON ${IBVERBS_LIBRARIES})
endif()
set(TEST_INC_COMMON PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
set(TEST_INC_INTERNAL PRIVATE ${PROJECT_SOURCE_DIR}/src/core/include)

if(MSCCLPP_USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
    foreach(arch ${MSCCLPP_GPU_ARCHS})
        add_compile_options(--offload-arch=${arch})
    endforeach()
endif()

function(add_test_executable name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${TEST_LIBS_COMMON} MPI::MPI_CXX)
    if(MSCCLPP_USE_IB)
        target_compile_definitions(${name} PRIVATE USE_IBVERBS)
    endif()
    target_include_directories(${name} ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
    target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
    add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_mpi_test.sh ${name} 2)
endfunction()

add_test_executable(allgather_test_cpp allgather_test_cpp.cu)
add_test_executable(allgather_test_host_offloading allgather_test_host_offloading.cu)
add_test_executable(nvls_test nvls_test.cu)
add_test_executable(executor_test executor_test.cc)

configure_file(run_mpi_test.sh.in run_mpi_test.sh)

include(CTest)

# Build test framework library
add_library(test_framework STATIC framework.cc)
target_include_directories(test_framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_framework PUBLIC MPI::MPI_CXX)

# Unit tests
add_executable(unit_tests)
target_link_libraries(unit_tests ${TEST_LIBS_COMMON} test_framework)
target_include_directories(unit_tests ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
add_subdirectory(unit)
add_test(NAME unit_tests COMMAND unit_tests)

# Multi-process unit tests
add_executable(mp_unit_tests)
target_link_libraries(mp_unit_tests ${TEST_LIBS_COMMON} test_framework MPI::MPI_CXX)
target_include_directories(mp_unit_tests ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
add_subdirectory(mp_unit)
add_test(NAME mp_unit_tests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_mpi_test.sh mp_unit_tests 2)

# mscclpp-test
add_subdirectory(mscclpp-test)

# Performance tests
add_subdirectory(perf)
