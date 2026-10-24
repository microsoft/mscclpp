# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Fetches and builds the third-party DOCA GPUNetIO (GDAKI) sources used by the GPU-initiated IB endpoint mode
# (EndpointConfig::Ib::Mode::GpuNetIo). See docs/guide/gpunetio.md.
#
# Defines:
#   mscclpp_gpunetio_headers  INTERFACE target with the GPUNetIO include directories.
#   mscclpp_gpunetio_obj      OBJECT library with the GPUNetIO host sources, built with relaxed warnings.

include(FetchContent)
FetchContent_Declare(gpunetio
    GIT_REPOSITORY https://github.com/NVIDIA-DOCA/gpunetio.git
    GIT_TAG b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b  # v5.0.0-rc1
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(gpunetio)
set(GPUNETIO_HOME ${gpunetio_SOURCE_DIR})

# In the CQ creation error path of doca_gpunetio_high_level.cpp, the per-QP cleanup frees the host staging buffer and
# the common exit path frees it again. Compile a copy of the file that clears the pointer after the first free. The
# fetched tree is left unmodified. Drop this when moving to a revision that fixes it upstream.
set(GPUNETIO_HIGH_LEVEL_UPSTREAM ${GPUNETIO_HOME}/src/doca_gpunetio_high_level.cpp)
set(GPUNETIO_HIGH_LEVEL_PATCHED ${CMAKE_BINARY_DIR}/gpunetio/patched/doca_gpunetio_high_level.cpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GPUNETIO_HIGH_LEVEL_UPSTREAM})
file(READ ${GPUNETIO_HIGH_LEVEL_UPSTREAM} GPUNETIO_HIGH_LEVEL_CONTENT)
set(GPUNETIO_CQ_DOUBLE_FREE [=[
        if (cq_ring_haddr) {
            free(cq_ring_haddr);
        }]=])
set(GPUNETIO_CQ_CLEANUP_FIX [=[
        if (cq_ring_haddr) {
            free(cq_ring_haddr);
            cq_ring_haddr = nullptr;
        }]=])
string(FIND "${GPUNETIO_HIGH_LEVEL_CONTENT}" "${GPUNETIO_CQ_DOUBLE_FREE}" GPUNETIO_CQ_FIRST)
string(FIND "${GPUNETIO_HIGH_LEVEL_CONTENT}" "${GPUNETIO_CQ_DOUBLE_FREE}" GPUNETIO_CQ_LAST REVERSE)
if(GPUNETIO_CQ_FIRST EQUAL -1 OR NOT GPUNETIO_CQ_FIRST EQUAL GPUNETIO_CQ_LAST)
    message(FATAL_ERROR "GPUNetIO: CQ cleanup fix does not apply to "
                        "${GPUNETIO_HIGH_LEVEL_UPSTREAM}; re-evaluate it when changing the pinned revision")
endif()
string(REPLACE "${GPUNETIO_CQ_DOUBLE_FREE}" "${GPUNETIO_CQ_CLEANUP_FIX}" GPUNETIO_HIGH_LEVEL_CONTENT
               "${GPUNETIO_HIGH_LEVEL_CONTENT}")
file(WRITE ${GPUNETIO_HIGH_LEVEL_PATCHED}.tmp "${GPUNETIO_HIGH_LEVEL_CONTENT}")
file(COPY_FILE ${GPUNETIO_HIGH_LEVEL_PATCHED}.tmp ${GPUNETIO_HIGH_LEVEL_PATCHED} ONLY_IF_DIFFERENT)
file(REMOVE ${GPUNETIO_HIGH_LEVEL_PATCHED}.tmp)

add_library(mscclpp_gpunetio_headers INTERFACE)
target_include_directories(mscclpp_gpunetio_headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${GPUNETIO_HOME}/include>
    $<BUILD_INTERFACE:${GPUNETIO_HOME}/include/host>)

add_library(mscclpp_gpunetio_obj OBJECT
    ${GPUNETIO_HOME}/src/doca_verbs_dev.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_qp.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_cq.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_device_attr.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_umem.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_srq.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_uar.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_cuda_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_log.cpp
    ${GPUNETIO_HIGH_LEVEL_PATCHED}
    ${GPUNETIO_HOME}/src/doca_gpunetio_signal_map.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_utils.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_mlx5dv_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_ibv_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_gdrcopy.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_sdk_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_dev_sdk_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_umem_sdk_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_uar_sdk_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_cq_sdk_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_qp_sdk_wrapper.cpp
)
target_link_libraries(mscclpp_gpunetio_obj PUBLIC mscclpp_gpunetio_headers)
target_include_directories(mscclpp_gpunetio_obj PRIVATE ${GPUNETIO_HOME}/src)
target_include_directories(mscclpp_gpunetio_obj SYSTEM PRIVATE ${GPU_INCLUDE_DIRS} ${IBVERBS_INCLUDE_DIRS}
    ${MLX5_INCLUDE_DIRS})
# Third-party code: do not enforce MSCCL++ warning flags.
target_compile_options(mscclpp_gpunetio_obj PRIVATE -w)
# Select the dlopen-based ibverbs/mlx5dv wrappers.
target_compile_definitions(mscclpp_gpunetio_obj PRIVATE DOCA_VERBS_USE_NET_WRAPPER)
set_target_properties(mscclpp_gpunetio_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1)

install(DIRECTORY ${GPUNETIO_HOME}/include/ DESTINATION include/mscclpp/gpunetio)
install(FILES ${GPUNETIO_HOME}/LICENSE.txt DESTINATION share/licenses/mscclpp/gpunetio)
