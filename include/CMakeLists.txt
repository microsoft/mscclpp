# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp ${CMAKE_CURRENT_BINARY_DIR}/*.hpp)
target_sources(
    mscclpp_obj PUBLIC FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    FILES ${HEADERS}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mscclpp/version.hpp DESTINATION include/mscclpp)
