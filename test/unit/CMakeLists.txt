# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

target_sources(unit_tests PRIVATE
    unit_tests_main.cc
    core_tests.cc
    gdr_tests.cu
    gpu_utils_tests.cc
    errors_tests.cc
    fifo_tests.cu
    fifo_perf_tests.cu
    numa_tests.cc
    socket_tests.cc
    utils_tests.cc
    utils_internal_tests.cc
    compile_tests.cu
    gpu_data_types_tests.cu
    local_channel_tests.cu
    gpu_ipc_mem_tests.cc
    bulk_tests.cu
)

# The production HCA selector is independent of CUDA/verbs and can be tested
# without allocating GPUs, opening HCAs, or launching MPI ranks.
add_executable(gpunetio_topology_test gpunetio_topology_test.cc)
target_include_directories(gpunetio_topology_test PRIVATE ${PROJECT_SOURCE_DIR}/src/gpunetio/host)
target_compile_features(gpunetio_topology_test PRIVATE cxx_std_17)
add_test(NAME gpunetio_topology_test COMMAND gpunetio_topology_test)
