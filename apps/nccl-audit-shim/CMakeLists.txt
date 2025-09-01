# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*)

if(MSCCLPP_USE_ROCM)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()


add_library(mscclpp_audit_nccl_obj OBJECT)
target_sources(mscclpp_audit_nccl_obj PRIVATE ${SOURCES})

add_library(mscclpp_audit_nccl SHARED)
target_link_libraries(mscclpp_audit_nccl PUBLIC mscclpp_audit_nccl_obj)
set_target_properties(mscclpp_audit_nccl PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

install(TARGETS mscclpp_audit_nccl
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)