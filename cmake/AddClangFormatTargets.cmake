# Add targets to run clang-format

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    message(STATUS "Found clang-format: ${CLANG_FORMAT}")
    set(CLANG_FORMAT_FILE_TYPES *.h *.hpp *.c *.cc *.cpp *.cu)
    # Produce combinations of source directories and file types
    foreach(SOURCE_DIR ${CLANG_FORMAT_SOURCE_DIRS})
        foreach(FILE_TYPE ${CLANG_FORMAT_FILE_TYPES})
            list(APPEND CLANG_FORMAT_SOURCE_PATTERNS ${SOURCE_DIR}/${FILE_TYPE})
        endforeach()
    endforeach()
    file(GLOB_RECURSE CLANG_FORMAT_SOURCES ${CLANG_FORMAT_SOURCE_PATTERNS})
    add_custom_target(check-format ALL COMMAND ${CLANG_FORMAT} -style=file --dry-run ${CLANG_FORMAT_SOURCES})
    add_custom_target(format COMMAND ${CLANG_FORMAT} -style=file -i ${CLANG_FORMAT_SOURCES})
else()
    message(STATUS "clang-format not found.")
endif()
