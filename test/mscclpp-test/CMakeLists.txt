# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

function(add_mscclpp_test_executable name sources)
    if(MSCCLPP_USE_ROCM)
        set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
    endif()
    add_executable(${name} ${sources} common.cc)
    target_link_libraries(${name} ${TEST_LIBS_COMMON} MPI::MPI_CXX nlohmann_json::nlohmann_json)
    target_include_directories(${name} ${TEST_INC_COMMON} ${TEST_INC_INTERNAL})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/mscclpp-test")
endfunction()

add_mscclpp_test_executable(sendrecv_test_perf sendrecv_test.cu)
add_mscclpp_test_executable(allgather_test_perf allgather_test.cu)
add_mscclpp_test_executable(allreduce_test_perf allreduce_test.cu)
add_mscclpp_test_executable(alltoall_test_perf alltoall_test.cu)
add_mscclpp_test_executable(ep_low_latency_opt_signal_perf ep_low_latency_opt_signal_perf.cu)
target_sources(ep_low_latency_opt_signal_perf PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ext/ep/low_latency/dispatch.cu)
target_include_directories(ep_low_latency_opt_signal_perf PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ext/ep
    ${PROJECT_SOURCE_DIR}/src/ext/ep/include)
if(MSCCLPP_USE_CUDA)
    set(_mscclpp_ep_test_gpu_archs "")
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_arch STREQUAL "native")
            list(APPEND _mscclpp_ep_test_gpu_archs "${_arch}")
        else()
            string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
            if(_arch_num AND _arch_num GREATER_EQUAL 90)
                list(APPEND _mscclpp_ep_test_gpu_archs "${_arch}")
            endif()
        endif()
    endforeach()
    set_target_properties(ep_low_latency_opt_signal_perf PROPERTIES
        CUDA_ARCHITECTURES "${_mscclpp_ep_test_gpu_archs}")

    add_executable(tma_pipeline_perf tma_pipeline_perf.cu)
    target_link_libraries(tma_pipeline_perf PRIVATE ${GPU_LIBRARIES})
    target_include_directories(tma_pipeline_perf ${TEST_INC_COMMON})
    set_target_properties(tma_pipeline_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/mscclpp-test"
        CUDA_ARCHITECTURES "${_mscclpp_ep_test_gpu_archs}")
endif()
