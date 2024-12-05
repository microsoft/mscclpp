# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(MPI)

add_executable(nccl_api_test nccl_api_test.cc)
target_link_libraries(nccl_api_test mscclpp mscclpp_nccl ${GPU_LIBRARIES} ${NUMA_LIBRARIES} Threads::Threads MPI::MPI_CXX)
if(IBVERBS_FOUND)
    target_link_libraries(nccl_api_test ${IBVERBS_LIBRARIES})
    target_compile_definitions(nccl_api_test PRIVATE USE_IBVERBS)
endif()
target_include_directories(nccl_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/apps/nccl/include)
