add_test_executable(ib_test ib_test.cc)
target_link_libraries(ib_test CUDA::cudart)