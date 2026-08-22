# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(MSCCLPP_BUILD_EXT_COLLECTIVES)
    add_subdirectory(collectives)
endif()

if(MSCCLPP_BUILD_EXT_NCCL)
    add_subdirectory(nccl)
endif()

if(MSCCLPP_BUILD_EXT_EP)
    if(NOT MSCCLPP_BUILD_PYTHON_BINDINGS)
        message(STATUS "Skipping the EP extension because Python bindings are disabled.")
    elseif(MSCCLPP_USE_CUDA)
        add_subdirectory(ep)
    else()
        message(STATUS "Skipping the EP extension: only CUDA is currently supported.")
    endif()
endif()
