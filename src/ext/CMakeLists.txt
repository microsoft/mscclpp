# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(MSCCLPP_BUILD_EXT_COLLECTIVES)
    add_subdirectory(collectives)
endif()

if(MSCCLPP_BUILD_EXT_NCCL)
    add_subdirectory(nccl)
endif()

if(MSCCLPP_BUILD_EXT_EP)
    if(MSCCLPP_USE_CUDA)
        add_subdirectory(ep)
    else()
        message(STATUS "Skipping the EP extension: only CUDA is currently supported.")
    endif()
endif()
