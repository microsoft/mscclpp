# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# GPU detection and configuration

# GPU detection
if(MSCCLPP_BYPASS_GPU_CHECK)
    if(MSCCLPP_USE_CUDA)
        message(STATUS "Bypassing GPU check: using NVIDIA/CUDA")
        find_package(CUDAToolkit REQUIRED)
    elseif(MSCCLPP_USE_ROCM)
        message(STATUS "Bypassing GPU check: using AMD/ROCm")
        find_package(hip REQUIRED)
    else()
        message(FATAL_ERROR "Bypassing GPU check: neither NVIDIA/CUDA nor AMD/ROCm is specified")
    endif()
else()
    include(CheckNvidiaGpu)
    include(CheckAmdGpu)

    if(NVIDIA_FOUND AND AMD_FOUND)
        message(STATUS "Detected NVIDIA/CUDA and AMD/ROCm: prioritizing NVIDIA/CUDA")
        set(MSCCLPP_USE_CUDA ON)
        set(MSCCLPP_USE_ROCM OFF)
    elseif(NVIDIA_FOUND)
        message(STATUS "Detected NVIDIA/CUDA")
        set(MSCCLPP_USE_CUDA ON)
        set(MSCCLPP_USE_ROCM OFF)
    elseif(AMD_FOUND)
        message(STATUS "Detected AMD/ROCm")
        set(MSCCLPP_USE_CUDA OFF)
        set(MSCCLPP_USE_ROCM ON)
    else()
        message(FATAL_ERROR "Neither NVIDIA/CUDA nor AMD/ROCm is found")
    endif()
endif()

# GPU architectures configuration
if(MSCCLPP_GPU_ARCHS)
    string(STRIP "${MSCCLPP_GPU_ARCHS}" MSCCLPP_GPU_ARCHS)
    string(REPLACE " " ";" MSCCLPP_GPU_ARCHS "${MSCCLPP_GPU_ARCHS}")
    string(REPLACE "," ";" MSCCLPP_GPU_ARCHS "${MSCCLPP_GPU_ARCHS}")

    if(NOT MSCCLPP_GPU_ARCHS)
        message(FATAL_ERROR "MSCCLPP_GPU_ARCHS cannot be empty")
    endif()
elseif(MSCCLPP_USE_CUDA)
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    set(MSCCLPP_GPU_ARCHS 80)  # Ampere
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        list(APPEND MSCCLPP_GPU_ARCHS 90)  # Hopper
        if(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 8)
            list(APPEND MSCCLPP_GPU_ARCHS 100)  # Blackwell
        endif()
    endif()
elseif(MSCCLPP_USE_ROCM)
    set(CMAKE_HIP_ARCHITECTURES gfx90a gfx941 gfx942)
endif()

if(MSCCLPP_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${MSCCLPP_GPU_ARCHS})
    set(GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver)
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
elseif(MSCCLPP_USE_ROCM)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")
    set(CMAKE_HIP_ARCHITECTURES ${MSCCLPP_GPU_ARCHS})
    set(GPU_LIBRARIES hip::device)
    set(GPU_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
endif()

message(STATUS "GPU architectures: ${MSCCLPP_GPU_ARCHS}")
