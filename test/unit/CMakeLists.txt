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

foreach(mode IN ITEMS release debug)
    add_executable(gpunetio_channel_${mode}_test gpunetio_channel_test.cc)
    target_include_directories(gpunetio_channel_${mode}_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(mode STREQUAL "debug")
        target_compile_definitions(gpunetio_channel_${mode}_test PRIVATE TEST_DEVICE_ASSERTS)
    endif()
    add_test(NAME gpunetio_channel_${mode}_test COMMAND gpunetio_channel_${mode}_test)
endforeach()

add_executable(gpunetio_channel_host_test gpunetio_channel_test.cc)
target_include_directories(gpunetio_channel_host_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gpunetio_channel_host_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
target_compile_definitions(gpunetio_channel_host_test PRIVATE TEST_HOST_API)
if(MSCCLPP_USE_GPUNETIO)
    target_compile_definitions(gpunetio_channel_host_test PRIVATE TEST_GPUNETIO_ENABLED)
endif()
target_link_libraries(gpunetio_channel_host_test PRIVATE mscclpp)
add_test(NAME gpunetio_channel_host_test COMMAND gpunetio_channel_host_test)
