# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS include/nccl.h)

if(USE_ROCM)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(mscclpp_nccl_obj OBJECT)
target_sources(mscclpp_nccl_obj PRIVATE ${SOURCES})
target_sources(mscclpp_nccl_obj PUBLIC FILE_SET HEADERS FILES ${HEADERS})
target_include_directories(mscclpp_nccl_obj PRIVATE ${GPU_INCLUDE_DIRS} include)
target_link_libraries(mscclpp_nccl_obj PRIVATE ${GPU_LIBRARIES} PUBLIC mscclpp_obj)
set_target_properties(mscclpp_nccl_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
if(USE_CUDA)
    target_compile_definitions(mscclpp_nccl_obj PRIVATE USE_CUDA)
elseif(USE_ROCM)
    target_compile_definitions(mscclpp_nccl_obj PRIVATE USE_ROCM)
endif()

add_library(mscclpp_nccl SHARED)
target_link_libraries(mscclpp_nccl PUBLIC mscclpp_nccl_obj)
set_target_properties(mscclpp_nccl PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclpp_nccl_static STATIC)
target_link_libraries(mscclpp_nccl_static PUBLIC mscclpp_nccl_obj)
set_target_properties(mscclpp_nccl_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

install(TARGETS mscclpp_nccl_obj
    FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
install(TARGETS mscclpp_nccl
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_nccl_static
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
