add_executable(bootstrap_test_cpp bootstrap_test_cpp.cc)
target_link_libraries(bootstrap_test_cpp mscclpp MPI::MPI_CXX)

add_executable(communicator_test_cpp communicator_test_cpp.cu)
target_link_libraries(communicator_test_cpp mscclpp MPI::MPI_CXX)

add_executable(allgather_test_cpp allgather_test_cpp.cu)
target_link_libraries(allgather_test_cpp mscclpp MPI::MPI_CXX)

add_subdirectory(unittests)
