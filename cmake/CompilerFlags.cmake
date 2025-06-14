# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Compiler flags configuration

# Compiler and language configuration
set(CMAKE_CXX_STANDARD 17)

# Base compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# GCC-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wduplicated-cond -Wduplicated-branches")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op -Wnull-dereference")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuseless-cast -Wdouble-promotion -Wformat=2")
endif()

# Clang-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-compat -Wno-pre-c++14-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors -Wno-global-constructors")
endif()

# Treat warnings as errors if requested
if(MSCCLPP_TREAT_WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_BUILD)
endif()
