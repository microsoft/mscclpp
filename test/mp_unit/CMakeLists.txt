# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(mp_unit_tests PRIVATE
    mp_unit_tests.cc
    bootstrap_tests.cc
    ib_tests.cu
    communicator_tests.cu
    port_channel_tests.cu
    memory_channel_tests.cu
    bulk_pattern_tests.cu
    semaphore_perf_tests.cu
    switch_channel_tests.cu
    switch_channel_perf_tests.cu
    executor_tests.cc
)

if(TARGET mscclpp_ep)
    target_sources(mp_unit_tests PRIVATE ep_tests.cu)
    target_include_directories(mp_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/ext/ep/include)
    target_link_libraries(mp_unit_tests mscclpp_ep)
endif()
