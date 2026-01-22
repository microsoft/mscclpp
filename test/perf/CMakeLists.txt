# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find required packages
find_package(MPI REQUIRED)

# Note: nlohmann_json::nlohmann_json target is already available from the main project

# Set up common libraries and includes for tests
set(PERF_TEST_LIBS_COMMON mscclpp ${GPU_LIBRARIES} ${NUMA_LIBRARIES} Threads::Threads MPI::MPI_CXX)
if(MSCCLPP_USE_IB)
    list(APPEND PERF_TEST_LIBS_COMMON ${IBVERBS_LIBRARIES})
endif()

set(PERF_TEST_INC_COMMON 
    PRIVATE ${PROJECT_SOURCE_DIR}/include 
    SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})

# Function to add a test executable
function(add_perf_test_executable name sources)
    if(MSCCLPP_USE_ROCM)
        set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
    endif()
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${PERF_TEST_LIBS_COMMON})
    
    # Link nlohmann_json - use the target from main project
    target_link_libraries(${name} nlohmann_json::nlohmann_json)
    
    if(MSCCLPP_USE_IB)
        target_compile_definitions(${name} PRIVATE USE_IBVERBS)
    endif()
    
    target_include_directories(${name} ${PERF_TEST_INC_COMMON})
    target_compile_definitions(${name} PRIVATE MSCCLPP_USE_MPI_FOR_TESTS)
    
    # Set C++ standard
    target_compile_features(${name} PRIVATE cxx_std_17)

    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/perf")
endfunction()

# Add FIFO test  
add_perf_test_executable(fifo_test "framework.cc;fifo_test.cu")
