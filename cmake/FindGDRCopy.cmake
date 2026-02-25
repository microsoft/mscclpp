# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Find the GDRCopy libraries
#
# The following variables are optionally searched for defaults
#  GDRCOPY_ROOT_DIR: Base directory where all GDRCopy components are found
#  GDRCOPY_INCLUDE_DIR: Directory where GDRCopy headers are found
#  GDRCOPY_LIB_DIR: Directory where GDRCopy libraries are found

# The following are set after configuration is done:
#  GDRCOPY_FOUND
#  GDRCOPY_INCLUDE_DIRS
#  GDRCOPY_LIBRARIES

find_path(GDRCOPY_INCLUDE_DIRS
  NAMES gdrapi.h
  HINTS
  ${GDRCOPY_INCLUDE_DIR}
  ${GDRCOPY_ROOT_DIR}
  ${GDRCOPY_ROOT_DIR}/include
  /usr/local/include
  /usr/include)

find_library(GDRCOPY_LIBRARIES
  NAMES gdrapi
  HINTS
  ${GDRCOPY_LIB_DIR}
  ${GDRCOPY_ROOT_DIR}
  ${GDRCOPY_ROOT_DIR}/lib
  /usr/local/lib
  /usr/lib
  /usr/lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDRCopy DEFAULT_MSG GDRCOPY_INCLUDE_DIRS GDRCOPY_LIBRARIES)
mark_as_advanced(GDRCOPY_INCLUDE_DIRS GDRCOPY_LIBRARIES)
