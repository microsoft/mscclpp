# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(mp_unit_tests PRIVATE
    mp_unit_tests.cc
    bootstrap_tests.cc
    ib_tests.cu
    communicator_tests.cu
    device_channel_tests.cu
    direct_channel_tests.cu
)
