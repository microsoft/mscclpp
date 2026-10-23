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
    GIT_TAG bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4  # v4.0.1
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(gpunetio)
set(GPUNETIO_HOME ${gpunetio_SOURCE_DIR})

# Backport from upstream v5.0.0-rc1 (b2b7a291ec70): in v4.0.1, doca_gpu_verbs_create_qp_hl returns directly when
# doca_gpu_verbs_export_qp fails, leaking the CQ/UAR/QP it already created. Compile a copy of the file that takes the
# existing cleanup path instead. The fetched tree is left unmodified. Drop this when moving to a fixed revision.
set(GPUNETIO_HIGH_LEVEL_UPSTREAM ${GPUNETIO_HOME}/src/doca_gpunetio_high_level.cpp)
set(GPUNETIO_HIGH_LEVEL_PATCHED ${CMAKE_BINARY_DIR}/gpunetio/patched/doca_gpunetio_high_level.cpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GPUNETIO_HIGH_LEVEL_UPSTREAM})
file(READ ${GPUNETIO_HIGH_LEVEL_UPSTREAM} GPUNETIO_HIGH_LEVEL_CONTENT)
set(GPUNETIO_EXPORT_QP_LEAK [=[
                                      enable_data_direct, &qp_->qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        return status;
    }]=])
set(GPUNETIO_EXPORT_QP_FIX [=[
                                      enable_data_direct, &qp_->qp_gverbs);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create GPU verbs QP");
        goto exit_error;
    }]=])
string(FIND "${GPUNETIO_HIGH_LEVEL_CONTENT}" "${GPUNETIO_EXPORT_QP_LEAK}" GPUNETIO_LEAK_FIRST)
string(FIND "${GPUNETIO_HIGH_LEVEL_CONTENT}" "${GPUNETIO_EXPORT_QP_LEAK}" GPUNETIO_LEAK_LAST REVERSE)
if(GPUNETIO_LEAK_FIRST EQUAL -1 OR NOT GPUNETIO_LEAK_FIRST EQUAL GPUNETIO_LEAK_LAST)
    message(FATAL_ERROR "GPUNetIO: doca_gpu_verbs_create_qp_hl export-failure backport does not apply to "
                        "${GPUNETIO_HIGH_LEVEL_UPSTREAM}; re-evaluate it when changing the pinned revision")
endif()
string(REPLACE "${GPUNETIO_EXPORT_QP_LEAK}" "${GPUNETIO_EXPORT_QP_FIX}" GPUNETIO_HIGH_LEVEL_CONTENT
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
    ${GPUNETIO_HOME}/src/doca_gpunetio_log.cpp
    ${GPUNETIO_HIGH_LEVEL_PATCHED}
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
