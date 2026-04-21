# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

add_subdirectory(csrc)
add_subdirectory(test)

target_compile_definitions(mscclpp_py PRIVATE
  $<$<BOOL:${MSCCLPP_DISABLE_NB_LEAK_WARNINGS}>:MSCCLPP_DISABLE_NB_LEAK_WARNINGS>
)

add_custom_target(pytest_lib_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_mscclpp.*.so
        ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_ext.*.so
        ${CMAKE_CURRENT_SOURCE_DIR}/test/_cpp
    DEPENDS mscclpp_py mscclpp_py_test
)