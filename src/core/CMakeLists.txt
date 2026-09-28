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
    if(MLX5_FOUND)
        target_include_directories(mscclpp_obj SYSTEM PRIVATE ${MLX5_INCLUDE_DIRS})
        target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_MLX5DV)
    endif()
endif()

if(MSCCLPP_USE_GDRCOPY)
    target_include_directories(mscclpp_obj SYSTEM PRIVATE ${GDRCOPY_INCLUDE_DIRS})
    target_link_libraries(mscclpp_obj PRIVATE ${GDRCOPY_LIBRARIES})
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_GDRCOPY)
endif()

if(MSCCLPP_USE_GPUNETIO)
    # GPU-initiated networking backend for PortChannel (DOCA GPUNetIO dependency).
    # Link the host object library so the host-side GpuNetIo service can
    # create QPs / register MRs. Consumer usage requirements are attached to the
    # final shared/static targets below, independently of internal object compilation.
    target_link_libraries(mscclpp_obj PRIVATE mscclpp_gpunetio_obj)
    target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_HAS_GPUNETIO)
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

if(MSCCLPP_USE_MRC)
	target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_MRC)
endif()

# libmscclpp
add_library(mscclpp SHARED)
target_link_libraries(mscclpp PUBLIC $<BUILD_INTERFACE:mscclpp_obj>)
if(MSCCLPP_USE_GPUNETIO)
    target_link_libraries(mscclpp PUBLIC $<BUILD_INTERFACE:mscclpp_gpunetio_obj>)
endif()
set_target_properties(mscclpp PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

add_library(mscclpp_static STATIC)
target_link_libraries(mscclpp_static PUBLIC $<BUILD_INTERFACE:mscclpp_obj>)
if(MSCCLPP_USE_GPUNETIO)
    target_link_libraries(mscclpp_static PUBLIC $<BUILD_INTERFACE:mscclpp_gpunetio_obj>)
endif()
set_target_properties(mscclpp_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

foreach(target IN ITEMS mscclpp mscclpp_static)
    target_include_directories(${target} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(${target} INTERFACE cxx_std_20)
    if(MSCCLPP_USE_GPUNETIO)
        target_link_libraries(${target} INTERFACE mscclpp_gpunetio_headers)
        target_compile_definitions(${target} INTERFACE MSCCLPP_USE_GPUNETIO)
    endif()
endforeach()
add_library(mscclpp::mscclpp ALIAS mscclpp)
add_library(mscclpp::mscclpp_static ALIAS mscclpp_static)

install(TARGETS mscclpp EXPORT mscclppTargets
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_static EXPORT mscclppTargets
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)
install(EXPORT mscclppTargets NAMESPACE mscclpp:: DESTINATION lib/cmake/mscclpp)
include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/mscclpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mscclpp-config.cmake INSTALL_DESTINATION lib/cmake/mscclpp)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mscclpp-config.cmake
    ${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake ${PROJECT_SOURCE_DIR}/cmake/FindIBVerbs.cmake
    ${PROJECT_SOURCE_DIR}/cmake/FindGDRCopy.cmake DESTINATION lib/cmake/mscclpp)
