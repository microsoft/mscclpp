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
