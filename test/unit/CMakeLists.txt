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
target_include_directories(gpunetio_channel_host_test PRIVATE ${PROJECT_SOURCE_DIR}/src/gpunetio/host)
target_include_directories(gpunetio_channel_host_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
target_compile_definitions(gpunetio_channel_host_test PRIVATE TEST_HOST_API)
if(MSCCLPP_USE_GPUNETIO)
    target_compile_definitions(gpunetio_channel_host_test PRIVATE TEST_GPUNETIO_ENABLED)
endif()
target_link_libraries(gpunetio_channel_host_test PRIVATE mscclpp Threads::Threads)
add_test(NAME gpunetio_channel_host_test COMMAND gpunetio_channel_host_test)

if(MSCCLPP_USE_GPUNETIO)
    set(_setup_source ${PROJECT_SOURCE_DIR}/src/gpunetio/host/gpu_net_io_service.cpp)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_setup_source})
    file(READ ${_setup_source} _setup_text)
    string(FIND "${_setup_text}" "struct QpExchangeInfo {" _metadata_start)
    string(FIND "${_setup_text}" "}  // namespace" _metadata_end)
    string(FIND "${_setup_text}" "void GpuNetIoService::setupImpl(" _setup_start)
    string(FIND "${_setup_text}" "\nMSCCLPP_API_CPP GpuNetIoDeviceContext*" _setup_end)
    if(_metadata_start LESS 0 OR _metadata_end LESS _metadata_start OR _setup_start LESS 0 OR _setup_end LESS _setup_start)
        message(FATAL_ERROR "Cannot locate production GPUNetIO setup for CPU regression")
    endif()
    math(EXPR _metadata_size "${_metadata_end} - ${_metadata_start}")
    math(EXPR _setup_size "${_setup_end} - ${_setup_start}")
    string(SUBSTRING "${_setup_text}" ${_metadata_start} ${_metadata_size} _metadata)
    string(SUBSTRING "${_setup_text}" ${_setup_start} ${_setup_size} _setup_body)
    string(REPLACE "void GpuNetIoService::setupImpl" "void State::setupImpl" _setup_body "${_setup_body}")
    string(REPLACE "auto& s = *pimpl_;" "auto& s = *this;" _setup_body "${_setup_body}")
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpunetio_setup_metadata.inc CONTENT "${_metadata}")
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpunetio_setup_body.inc CONTENT "${_setup_body}")
    add_executable(gpunetio_setup_test gpunetio_setup_test.cc)
    target_include_directories(gpunetio_setup_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/src/gpunetio/host)
    target_include_directories(gpunetio_setup_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS} ${IBVERBS_INCLUDE_DIRS})
    target_link_libraries(gpunetio_setup_test PRIVATE mscclpp Threads::Threads)
    add_test(NAME gpunetio_setup_test COMMAND gpunetio_setup_test)
    set_tests_properties(gpunetio_setup_test PROPERTIES TIMEOUT 30)

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
    target_link_libraries(gpunetio_channel_qp_table_test PRIVATE mscclpp_gpunetio_headers Threads::Threads)
    target_include_directories(gpunetio_channel_qp_table_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS} ${IBVERBS_INCLUDE_DIRS})
    add_test(NAME gpunetio_channel_qp_table_test COMMAND gpunetio_channel_qp_table_test)

    foreach(mode IN ITEMS clean existing_macros)
        add_library(gpunetio_header_${mode}_compile OBJECT compile_tests.cu)
        target_compile_definitions(gpunetio_header_${mode}_compile PRIVATE TEST_GPUNETIO_HEADER)
        if(mode STREQUAL "existing_macros")
            target_compile_definitions(gpunetio_header_${mode}_compile PRIVATE TEST_EXISTING_UNROLL_MACROS)
        endif()
        target_link_libraries(gpunetio_header_${mode}_compile PRIVATE mscclpp)
    endforeach()
    add_library(gpunetio_header_static_compile OBJECT compile_tests.cu)
    target_compile_definitions(gpunetio_header_static_compile PRIVATE TEST_GPUNETIO_HEADER)
    target_link_libraries(gpunetio_header_static_compile PRIVATE mscclpp_static)
endif()
