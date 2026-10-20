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

if(TARGET mscclpp_ep)
    target_sources(unit_tests PRIVATE ep_quantization_tests.cu)
    target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/ext/ep ${PROJECT_SOURCE_DIR}/src/ext/ep/include)
endif()
