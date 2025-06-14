# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
nanobind_add_module(mscclpp_py ${SOURCES})
set_target_properties(mscclpp_py PROPERTIES OUTPUT_NAME _mscclpp)

# Suppress warnings from external headers if enabled
if(MSCCLPP_SUPPRESS_EXTERNAL_WARNINGS)
    # Add Python includes as system includes to suppress warnings
    target_include_directories(mscclpp_py SYSTEM PRIVATE
        ${Python_INCLUDE_DIRS}
        ${nanobind_SOURCE_DIR}/include
    )

    # Add GPU includes as SYSTEM if a GPU backend is active
    if(DEFINED GPU_INCLUDE_DIRS AND (MSCCLPP_USE_CUDA OR MSCCLPP_USE_ROCM))
        target_include_directories(mscclpp_py SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
    endif()
endif()

target_link_libraries(mscclpp_py PRIVATE dlpack mscclpp_static ${GPU_LIBRARIES})

install(TARGETS mscclpp_py LIBRARY DESTINATION .)
