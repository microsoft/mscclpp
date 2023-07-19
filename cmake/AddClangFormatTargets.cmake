# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Add targets to run clang-format

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    message(STATUS "Found clang-format: ${CLANG_FORMAT}")
    set(FIND_DIRS ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/python ${PROJECT_SOURCE_DIR}/test)
    add_custom_target(check-format ALL
        COMMAND ${CLANG_FORMAT} -style=file --dry-run `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -style=file -i `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
else()
    message(STATUS "clang-format not found.")
endif()
