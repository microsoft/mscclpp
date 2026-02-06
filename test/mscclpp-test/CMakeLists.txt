# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

# Include path for collective algorithm headers (alltoallv, etc.)
set(COLLECTIVES_INC_DIR ${PROJECT_SOURCE_DIR}/src/ext/collectives/include)

function(add_mscclpp_test_executable name sources)
    if(MSCCLPP_USE_ROCM)
        set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
    endif()
    add_executable(${name} ${sources} common.cc)
    target_link_libraries(${name} ${TEST_LIBS_COMMON} MPI::MPI_CXX nlohmann_json::nlohmann_json)
    target_include_directories(${name} ${TEST_INC_COMMON} ${TEST_INC_INTERNAL} PRIVATE ${COLLECTIVES_INC_DIR})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/mscclpp-test")
endfunction()

add_mscclpp_test_executable(sendrecv_test_perf sendrecv_test.cu)
add_mscclpp_test_executable(allgather_test_perf allgather_test.cu)
add_mscclpp_test_executable(allreduce_test_perf allreduce_test.cu)
add_mscclpp_test_executable(alltoall_test_perf alltoall_test.cu)
add_mscclpp_test_executable(alltoallv_test_perf alltoallv_test.cu)
