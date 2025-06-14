# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# External dependencies configuration

include(FetchContent)

# Configure warning suppression for external dependencies
function(setup_external_warning_suppression)
    if(MSCCLPP_SUPPRESS_EXTERNAL_WARNINGS)
        message(STATUS "External dependency warnings: Suppressed")
        # Set policy for treating external dependencies as system includes
        if(POLICY CMP0077)
            cmake_policy(SET CMP0077 NEW)
        endif()
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW PARENT_SCOPE)
    else()
        message(STATUS "External dependency warnings: Enabled")
    endif()

    set(FETCHCONTENT_QUIET ON PARENT_SCOPE)
endfunction()

# Function to mark external target includes as system
function(mark_as_system_include target_name)
    if(MSCCLPP_SUPPRESS_EXTERNAL_WARNINGS AND TARGET ${target_name})
        get_target_property(target_include_dirs ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
        if(target_include_dirs)
            set_target_properties(${target_name} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${target_include_dirs}"
            )
        endif()
    endif()
endfunction()

# Setup warning suppression
setup_external_warning_suppression()

# Find system dependencies
find_package(IBVerbs)
find_package(NUMA REQUIRED)
find_package(Threads REQUIRED)

# Declare external dependencies
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)

if(MSCCLPP_BUILD_TESTS)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
endif()

if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v1.4.0)
    FetchContent_Declare(dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG 5c210da409e7f1e51ddf445134a4376fdbd70d7d)
endif()

# Make dependencies available (exclude from installation)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

if(MSCCLPP_BUILD_TESTS)
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    FetchContent_GetProperties(nanobind)
    if(NOT nanobind_POPULATED)
        FetchContent_Populate(nanobind)
        add_subdirectory(${nanobind_SOURCE_DIR} ${nanobind_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_GetProperties(dlpack)
    if(NOT dlpack_POPULATED)
        FetchContent_Populate(dlpack)
        add_subdirectory(${dlpack_SOURCE_DIR} ${dlpack_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

# Mark external dependencies as system includes to suppress their warnings
mark_as_system_include(nlohmann_json)
if(MSCCLPP_BUILD_TESTS)
    mark_as_system_include(gtest)
    mark_as_system_include(gtest_main)
    mark_as_system_include(gmock)
    mark_as_system_include(gmock_main)
endif()

if(MSCCLPP_BUILD_PYTHON_BINDINGS)
    # Note: nanobind is header-only and doesn't create a library target
    # Python binding targets handle nanobind includes directly
    mark_as_system_include(dlpack)
endif()
