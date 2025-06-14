# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
nanobind_add_module(mscclpp_py_test ${SOURCES})
set_target_properties(mscclpp_py_test PROPERTIES OUTPUT_NAME _ext)

# Suppress warnings from external headers if enabled
if(MSCCLPP_SUPPRESS_EXTERNAL_WARNINGS)
    # Add Python and nanobind includes as system includes to suppress warnings
    target_include_directories(mscclpp_py_test SYSTEM PRIVATE
        ${Python_INCLUDE_DIRS}
        ${nanobind_SOURCE_DIR}/include # Assuming nanobind_SOURCE_DIR is available
    )
    if(DEFINED GPU_INCLUDE_DIRS)
        target_include_directories(mscclpp_py_test SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
    endif()
endif()

target_link_libraries(mscclpp_py_test PRIVATE mscclpp_static ${GPU_LIBRARIES})
