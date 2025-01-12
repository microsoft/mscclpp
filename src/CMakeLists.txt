# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cu)
target_sources(mscclpp_obj PRIVATE ${SOURCES})
target_include_directories(mscclpp_obj PRIVATE include)
