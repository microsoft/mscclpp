add_executable(ib_test ib_test.cc)
target_link_libraries(ib_test mscclpp MPI::MPI_CXX CUDA::cudart)