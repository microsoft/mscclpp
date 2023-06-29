# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp)
target_sources(mscclpp PUBLIC FILE_SET HEADERS FILES ${HEADERS})
install(TARGETS mscclpp FILE_SET HEADERS)
