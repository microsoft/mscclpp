add_executable(sendrecv_test_perf sendrecv_test.cu common.cu)
target_link_libraries(sendrecv_test_perf mscclpp MPI::MPI_CXX)