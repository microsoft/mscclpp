# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.9.2)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack.git
    GIT_TAG 5c210da409e7f1e51ddf445134a4376fdbd70d7d
)

FetchContent_GetProperties(dlpack)
if(NOT dlpack_POPULATED)
    FetchContent_Populate(dlpack)
    # Add dlpack subdirectory but exclude it from installation
    add_subdirectory(${dlpack_SOURCE_DIR} ${dlpack_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
nanobind_add_module(mscclpp_py ${SOURCES})
set_target_properties(mscclpp_py PROPERTIES OUTPUT_NAME _mscclpp)
set_target_properties(mscclpp_py PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
target_link_libraries(mscclpp_py PRIVATE dlpack mscclpp mscclpp_collectives ${GPU_LIBRARIES})
target_include_directories(mscclpp_py SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
install(TARGETS mscclpp_py LIBRARY DESTINATION .)
