function(add_test_executable name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} mscclpp CUDA::cudart CUDA::cuda_driver)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
    if(USE_MPI_FOR_TESTS)
        target_link_libraries(${name} MPI::MPI_CXX)
        target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
    endif()
endfunction()

add_test_executable(bootstrap_test_cpp bootstrap_test_cpp.cc)
add_test_executable(communicator_test_cpp communicator_test_cpp.cu)
add_test_executable(allgather_test_cpp allgather_test_cpp.cu)
add_test_executable(ib_test ib_test.cc)

# Unit tests
add_executable(unit_tests)
target_link_libraries(unit_tests GTest::gtest_main GTest::gmock_main mscclpp)
add_subdirectory(unit) # This adds the sources to the mscclpp target
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)