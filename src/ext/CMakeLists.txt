# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(MSCCLPP_BUILD_EXT_COLLECTIVES)
    add_subdirectory(collectives)
endif()

if(MSCCLPP_BUILD_EXT_NCCL)
    add_subdirectory(nccl)
endif()
