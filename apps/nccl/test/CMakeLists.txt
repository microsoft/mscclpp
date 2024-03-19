# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

add_executable(nccl_api_test nccl_api_test.cc)
target_link_libraries(nccl_api_test mscclpp ${GPU_LIBRARIES} ${NUMA_LIBRARIES} ${IBVERBS_LIBRARIES} Threads::Threads MPI::MPI_CXX)
target_include_directories(nccl_api_test ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/apps/nccl/include)
