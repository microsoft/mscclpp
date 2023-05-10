function(add_test_executable name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} mscclpp)
    if(USE_MPI_FOR_TESTS)
        target_link_libraries(${name} MPI::MPI_CXX)
        target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
    endif()
endfunction()

add_test_executable(bootstrap_test_cpp bootstrap_test_cpp.cc)
add_test_executable(communicator_test_cpp communicator_test_cpp.cu)
add_test_executable(allgather_test_cpp allgather_test_cpp.cu)

add_subdirectory(unittests)
