# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Find the native MRC headers and library.
#
# Optional search hints:
#  MRC_ROOT_DIR: Base directory of the MRC SDK
#  MRC_INCLUDE_DIR: Directory containing mrc.h
#  MRC_LIB_DIR: Directory containing libnv_mrc
#
# Sets MRC_FOUND, MRC_INCLUDE_DIRS and MRC_LIBRARIES.
# Creates the imported target MSCCLPP::mrc when found.

find_path(MRC_INCLUDE_DIRS
  NAMES mrc.h
  HINTS
  ${MRC_INCLUDE_DIR}
  ${MRC_ROOT_DIR}
  ${MRC_ROOT_DIR}/include)

find_library(MRC_LIBRARIES
  NAMES nv_mrc
  HINTS
  ${MRC_LIB_DIR}
  ${MRC_ROOT_DIR}
  ${MRC_ROOT_DIR}/lib
  ${MRC_ROOT_DIR}/lib/${CMAKE_LIBRARY_ARCHITECTURE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MRC DEFAULT_MSG MRC_INCLUDE_DIRS MRC_LIBRARIES)
mark_as_advanced(MRC_INCLUDE_DIRS MRC_LIBRARIES)

if(MRC_FOUND AND NOT TARGET MSCCLPP::mrc)
  add_library(MSCCLPP::mrc UNKNOWN IMPORTED)
  set_target_properties(MSCCLPP::mrc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MRC_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${MRC_LIBRARIES}")
endif()
