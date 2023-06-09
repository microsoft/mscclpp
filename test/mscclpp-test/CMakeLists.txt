function(add_mscclpp_test_executable name sources)
    add_executable(${name} ${sources} common.cu)
    target_link_libraries(${name} mscclpp MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver MSCCLPP::numa)
endfunction()

add_mscclpp_test_executable(sendrecv_test_perf sendrecv_test.cu)
add_mscclpp_test_executable(allgather_test_perf allgather_test.cu)
add_mscclpp_test_executable(allreduce_test_perf allreduce_test.cu)
add_mscclpp_test_executable(alltoall_test_perf alltoall_test.cu)
