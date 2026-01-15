# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cu *.cc *.cpp)
file(GLOB_RECURSE AUDIT_SHIM CONFIGURE_DEPENDS audit-shim/*)

if(MSCCLPP_USE_ROCM)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(mscclpp_nccl SHARED ${SOURCES})
target_include_directories(mscclpp_nccl PRIVATE 
    include 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(mscclpp_nccl PUBLIC mscclpp mscclpp_collectives)
target_link_libraries(mscclpp_nccl PRIVATE ${GPU_LIBRARIES})
set_target_properties(mscclpp_nccl PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_library(mscclpp_audit_nccl_obj OBJECT ${AUDIT_SHIM})
set_target_properties(mscclpp_audit_nccl_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclpp_audit_nccl SHARED)
target_link_libraries(mscclpp_audit_nccl PUBLIC mscclpp_audit_nccl_obj)
set_target_properties(mscclpp_audit_nccl PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})


if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_nccl PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_nccl PRIVATE MSCCLPP_USE_ROCM)
    foreach(arch ${MSCCLPP_GPU_ARCHS})
        target_compile_options(mscclpp_nccl PRIVATE --offload-arch=${arch})
    endforeach()
endif()
if(MSCCLPP_NPKIT_FLAGS)
    target_compile_definitions(mscclpp_nccl PRIVATE ${MSCCLPP_NPKIT_FLAGS})
endif()

install(TARGETS mscclpp_nccl
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_audit_nccl
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
