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

if(MSCCLPP_USE_GPUNETIO)
    add_executable(gpunetio_channel_multi_qp_test gpunetio_channel_test.cc)
    target_compile_definitions(gpunetio_channel_multi_qp_test PRIVATE TEST_DEVICE_QPS)
    target_include_directories(gpunetio_channel_multi_qp_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(gpunetio_channel_multi_qp_test PRIVATE mscclpp_gpunetio_headers)
    add_test(NAME gpunetio_channel_multi_qp_test COMMAND gpunetio_channel_multi_qp_test)

    add_executable(gpunetio_channel_qp_table_test gpunetio_channel_test.cc)
    target_compile_definitions(gpunetio_channel_qp_table_test PRIVATE TEST_QP_TABLE)
    target_include_directories(gpunetio_channel_qp_table_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/gpunetio/host)
    target_link_libraries(gpunetio_channel_qp_table_test PRIVATE mscclpp_gpunetio_headers)
    target_include_directories(gpunetio_channel_qp_table_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS} ${IBVERBS_INCLUDE_DIRS})
    add_test(NAME gpunetio_channel_qp_table_test COMMAND gpunetio_channel_qp_table_test)

    foreach(mode IN ITEMS clean existing_macros)
        add_library(gpunetio_header_${mode}_compile OBJECT compile_tests.cu)
        target_compile_definitions(gpunetio_header_${mode}_compile PRIVATE TEST_GPUNETIO_HEADER MSCCLPP_USE_GPUNETIO)
        if(mode STREQUAL "existing_macros")
            target_compile_definitions(gpunetio_header_${mode}_compile PRIVATE TEST_EXISTING_UNROLL_MACROS)
        endif()
        target_include_directories(gpunetio_header_${mode}_compile PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(gpunetio_header_${mode}_compile PRIVATE mscclpp_gpunetio_headers)
    endforeach()
endif()
