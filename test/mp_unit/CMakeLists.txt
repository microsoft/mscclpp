# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

target_sources(mp_unit_tests PRIVATE
    mp_unit_tests.cc
    bootstrap_tests.cc
    ib_tests.cu
    communicator_tests.cu
    port_channel_tests.cu
    memory_channel_tests.cu
    semaphore_perf_tests.cu
    switch_channel_tests.cu
    executor_tests.cc
)

# When the GPUNetIO (GDAKI) backend is built, compile the GpuNetIo point-to-point
# test path into port_channel_tests.cu. Scope the define + vendored include dirs
# to just this source so unrelated test TUs keep the trap device definitions.
if(MSCCLPP_USE_GPUNETIO)
    target_include_directories(mp_unit_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/src/gpunetio/include
        ${PROJECT_SOURCE_DIR}/src/gpunetio/include/host)
    target_compile_definitions(mp_unit_tests PRIVATE MSCCLPP_USE_GPUNETIO)
endif()
