add_executable(bootstrap_test bootstrap_test.cc)
target_link_libraries(bootstrap_test mscclpp MPI::MPI_CXX)

add_executable(allgather_test_cpp allgather_test_cpp.cu)
target_link_libraries(allgather_test_cpp mscclpp MPI::MPI_CXX)

add_executable(allgather_test_standalone allgather_test_standalone.cu)
target_link_libraries(allgather_test_standalone mscclpp MPI::MPI_CXX)
