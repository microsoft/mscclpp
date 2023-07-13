# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.4.0)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(mscclpp_py main_py.cpp core_py.cpp error_py.cpp proxy_channel_py.cpp)
set_target_properties(mscclpp_py PROPERTIES OUTPUT_NAME mscclpp)
target_link_libraries(mscclpp_py PRIVATE mscclpp_static)
target_include_directories(mscclpp_py PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
if (SKBUILD)
    install(TARGETS mscclpp_py LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
endif()
