# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(mp_unit_tests PRIVATE
    mp_unit_tests.cc
    bootstrap_tests.cc
    ib_tests.cu
    communicator_tests.cu
    port_channel_tests.cu
    memory_channel_tests.cu
    executor_tests.cc
    fifo_test_multi_gpu.cu
)
