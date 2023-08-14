# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

add_subdirectory(mscclpp)
add_subdirectory(test)

add_custom_target(copy-py)
add_custom_command(TARGET copy-py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/mscclpp/_mscclpp.cpython-38-x86_64-linux-gnu.so
        ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/test/_ext.cpython-38-x86_64-linux-gnu.so
        ${CMAKE_CURRENT_SOURCE_DIR}/test/_cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Copy python libraries"
)

