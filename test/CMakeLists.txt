function(add_test_executable name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} mscclpp CUDA::cudart CUDA::cuda_driver)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
    if(USE_MPI_FOR_TESTS)
        target_link_libraries(${name} MPI::MPI_CXX)
        target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
        add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/run_mpi_test.sh ${name} 2)
    endif()
endfunction()

add_test_executable(allgather_test_cpp allgather_test_cpp.cu)
add_test_executable(allgather_test_host_offloading allgather_test_host_offloading.cu)

configure_file(run_mpi_test.sh.in run_mpi_test.sh)

# Unit tests
add_executable(unit_tests)
target_link_libraries(unit_tests GTest::gtest_main GTest::gmock_main mscclpp CUDA::cudart CUDA::cuda_driver)
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
add_subdirectory(unit)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

# Multi-process unit tests
add_executable(mp_unit_tests)
target_link_libraries(mp_unit_tests mscclpp CUDA::cudart CUDA::cuda_driver MPI::MPI_CXX GTest::gtest_main GTest::gmock_main)
target_include_directories(mp_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
add_subdirectory(mp_unit)
gtest_discover_tests(mp_unit_tests DISCOVERY_MODE PRE_TEST)

# Msccclpp_test
add_subdirectory(mscclpp-test)
