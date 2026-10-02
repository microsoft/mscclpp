# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(NVIDIA_FOUND "FALSE")
set(NVIDIA_GPU_ARCH "")

find_package(CUDAToolkit)

if(NOT CUDAToolkit_FOUND)
    return()
endif()

set(CMAKE_CUDA_ARCHITECTURES native)
if(NOT CMAKE_CUDA_COMPILER)
    # In case the CUDA Toolkit directory is not in the PATH
    find_program(CUDA_COMPILER
                 NAMES nvcc
                 PATHS ${CUDAToolkit_BIN_DIR})
    if(NOT CUDA_COMPILER)
        message(WARNING "Could not find nvcc in ${CUDAToolkit_BIN_DIR}")
        unset(CMAKE_CUDA_ARCHITECTURES)
        return()
    endif()
    set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
endif()
enable_language(CUDA)

set(CHECK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_nvidia_gpu.cu")

try_run(RUN_RESULT COMPILE_SUCCESS
        SOURCES ${CHECK_SRC}
        RUN_OUTPUT_VARIABLE NVIDIA_GPU_ARCH_OUTPUT)

if(COMPILE_SUCCESS AND RUN_RESULT EQUAL 0)
    string(STRIP "${NVIDIA_GPU_ARCH_OUTPUT}" NVIDIA_GPU_ARCH_OUTPUT)
    if(NVIDIA_GPU_ARCH_OUTPUT MATCHES "^[0-9]+$")
        set(NVIDIA_FOUND "TRUE")
        set(NVIDIA_GPU_ARCH "${NVIDIA_GPU_ARCH_OUTPUT}")
    else()
        unset(CMAKE_CUDA_ARCHITECTURES)
    endif()
else()
    unset(CMAKE_CUDA_ARCHITECTURES)
endif()
