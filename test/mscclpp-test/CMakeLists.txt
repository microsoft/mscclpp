include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

function(add_mscclpp_test_executable name sources)
    add_executable(${name} ${sources} common.cc)
    target_link_libraries(${name} mscclpp MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver MSCCLPP::numa nlohmann_json::nlohmann_json)
endfunction()

add_mscclpp_test_executable(sendrecv_test_perf sendrecv_test.cu)
add_mscclpp_test_executable(allgather_test_perf allgather_test.cu)
add_mscclpp_test_executable(allreduce_test_perf allreduce_test.cu)
add_mscclpp_test_executable(alltoall_test_perf alltoall_test.cu)
