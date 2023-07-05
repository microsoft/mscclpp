# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc)
target_sources(mscclpp PRIVATE ${SOURCES})
target_include_directories(mscclpp PRIVATE include)
