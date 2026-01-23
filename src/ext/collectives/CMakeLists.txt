# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cu)

if(MSCCLPP_USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(mscclpp_collectives SHARED ${SOURCES})

target_include_directories(mscclpp_collectives PRIVATE
    include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${PROJECT_SOURCE_DIR}/src/ext/include
)

target_link_libraries(mscclpp_collectives PUBLIC mscclpp)
target_link_libraries(mscclpp_collectives PRIVATE ${GPU_LIBRARIES} ${NUMA_LIBRARIES} Threads::Threads)

set_target_properties(mscclpp_collectives PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_collectives PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_collectives PRIVATE MSCCLPP_USE_ROCM)
    foreach(arch ${MSCCLPP_GPU_ARCHS})
        target_compile_options(mscclpp_collectives PRIVATE --offload-arch=${arch})
    endforeach()
endif()

install(TARGETS mscclpp_collectives
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
