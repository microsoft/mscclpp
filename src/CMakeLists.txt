# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc)
target_sources(mscclppObj PRIVATE ${SOURCES})
target_include_directories(mscclppObj PRIVATE include)
