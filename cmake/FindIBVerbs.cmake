# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find the IB Verbs libraries
#
# The following variables are optionally searched for defaults
#  IBVERBS_ROOT_DIR: Base directory where all ibverbs components are found
#  IBVERBS_INCLUDE_DIR: Directory where ibverbs headers are found
#  IBVERBS_LIB_DIR: Directory where ibverbs libraries are found

# The following are set after configuration is done:
#  IBVERBS_FOUND
#  IBVERBS_INCLUDE_DIR
#  IBVERBS_LIBRARY

find_path(IBVERBS_INCLUDE_DIR
  NAMES infiniband/verbs.h
  HINTS
  ${IBVERBS_INCLUDE_DIR}
  ${IBVERBS_ROOT_DIR}
  ${IBVERBS_ROOT_DIR}/include)

find_library(IBVERBS_LIBRARY
  NAMES ibverbs
  HINTS
  ${IBVERBS_LIB_DIR}
  ${IBVERBS_ROOT_DIR}
  ${IBVERBS_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IBVerbs DEFAULT_MSG IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)
mark_as_advanced(IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)
