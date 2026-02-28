# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cu)

# Handle ROCm specific source property
if(MSCCLPP_USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()


add_library(mscclpp_obj OBJECT ${SOURCES})

target_include_directories(mscclpp_obj
    SYSTEM PRIVATE
    ${GPU_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS})

target_include_directories(mscclpp_obj PRIVATE 
    include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mscclpp_obj PRIVATE ${GPU_LIBRARIES} ${NUMA_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads dl)

if(MSCCLPP_USE_IB)
    target_include_directories(mscclpp_obj SYSTEM PRIVATE ${IBVERBS_INCLUDE_DIRS})
    target_link_libraries(mscclpp_obj PRIVATE ${IBVERBS_LIBRARIES})
    target_compile_definitions(mscclpp_obj PUBLIC USE_IBVERBS)
endif()

if(MSCCLPP_USE_GDRCOPY)
    target_include_directories(mscclpp_obj SYSTEM PRIVATE ${GDRCOPY_INCLUDE_DIRS})
    target_link_libraries(mscclpp_obj PRIVATE ${GDRCOPY_LIBRARIES})
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_GDRCOPY)
endif()

set_target_properties(mscclpp_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_ROCM)
    foreach(arch ${MSCCLPP_GPU_ARCHS})
        target_compile_options(mscclpp_obj PRIVATE --offload-arch=${arch})
    endforeach()
endif()

if(MSCCLPP_ENABLE_TRACE)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_ENABLE_TRACE)
endif()

if(MSCCLPP_NPKIT_FLAGS)
    target_compile_definitions(mscclpp_obj PRIVATE ${MSCCLPP_NPKIT_FLAGS})
endif()

# libmscclpp
add_library(mscclpp SHARED)
target_link_libraries(mscclpp PUBLIC mscclpp_obj)
set_target_properties(mscclpp PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_library(mscclpp_static STATIC)
target_link_libraries(mscclpp_static PUBLIC mscclpp_obj)
set_target_properties(mscclpp_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

install(TARGETS mscclpp
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_static
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)
