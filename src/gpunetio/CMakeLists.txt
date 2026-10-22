# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DOCA GPUNetIO (GDAKI) sources from the pinned NVIDIA-DOCA/gpunetio repository.
# Provides GPU-initiated networking (kernel-issued RDMA WRITE/atomic) used by
# the GPUNetIO PortChannel backend.
#
# Built as an OBJECT library with relaxed warnings (third-party code) and
# linked into mscclpp_obj.

set(GPUNETIO_HOME ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(gpunetio
    GIT_REPOSITORY https://github.com/NVIDIA-DOCA/gpunetio.git
    GIT_TAG bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(gpunetio)
set(GPUNETIO_VENDOR_HOME ${gpunetio_SOURCE_DIR})

# Backport from upstream v5.0.0-rc1 (b2b7a291ec70): in v4.0.1, doca_gpu_verbs_create_qp_hl returns directly when
# doca_gpu_verbs_export_qp fails, leaking the CQ/UAR/QP it already created and leaving the output handle null.
# Compile a copy of the file that takes the existing cleanup path instead. The fetched tree is left unmodified, so
# this also applies when FETCHCONTENT_SOURCE_DIR_GPUNETIO points at an external checkout.
set(GPUNETIO_HIGH_LEVEL_UPSTREAM ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio_high_level.cpp)
set(GPUNETIO_HIGH_LEVEL_PATCHED ${CMAKE_CURRENT_BINARY_DIR}/patched/doca_gpunetio_high_level.cpp)
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

message(STATUS "GPUNetIO: direct GPU doorbells and GPU-resident CQs required; CPU-assisted fallback disabled")

add_library(mscclpp_gpunetio_headers INTERFACE)
target_include_directories(mscclpp_gpunetio_headers INTERFACE
    $<BUILD_INTERFACE:${GPUNETIO_VENDOR_HOME}/include>
    $<BUILD_INTERFACE:${GPUNETIO_VENDOR_HOME}/include/host>
    $<INSTALL_INTERFACE:include/mscclpp/gpunetio>)

set(GPUNETIO_SOURCES
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_dev.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_qp.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_cq.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_device_attr.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_umem.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_srq.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_uar.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio_log.cpp
    ${GPUNETIO_HIGH_LEVEL_PATCHED}
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_mlx5dv_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_ibv_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio_gdrcopy.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio_sdk_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_dev_sdk_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_umem_sdk_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_uar_sdk_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_cq_sdk_wrapper.cpp
    ${GPUNETIO_VENDOR_HOME}/src/doca_verbs_qp_sdk_wrapper.cpp
)

# MSCCL++ host service that drives the DOCA host API to bring up the
# GPU-initiated PortChannel backend (QP create/connect, MR registration).
set(GPUNETIO_HOST_SOURCES
    ${GPUNETIO_HOME}/host/gpu_net_io_service.cpp
)

add_library(mscclpp_gpunetio_obj OBJECT ${GPUNETIO_SOURCES} ${GPUNETIO_HOST_SOURCES})

target_include_directories(mscclpp_gpunetio_obj
    SYSTEM PRIVATE
    ${GPU_INCLUDE_DIRS}
    ${IBVERBS_INCLUDE_DIRS}
    ${MLX5_INCLUDE_DIRS})

target_link_libraries(mscclpp_gpunetio_obj PUBLIC mscclpp_gpunetio_headers)

target_include_directories(mscclpp_gpunetio_obj PRIVATE
    ${GPUNETIO_VENDOR_HOME}/src
    ${GPUNETIO_HOME}
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include)

# Third-party code: do not enforce mscclpp's -Wall -Wextra -Werror.
set_source_files_properties(${GPUNETIO_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")

# DOCA_VERBS_USE_NET_WRAPPER selects the dlopen-based ibverbs/mlx5dv wrappers.
target_compile_definitions(mscclpp_gpunetio_obj PRIVATE DOCA_VERBS_USE_NET_WRAPPER)

# The host service uses mscclpp core's IbCtx/IbMr (real interface guarded by
# USE_IBVERBS) for the ibverbs context / pd / MR registration.
if(MSCCLPP_USE_IB)
    target_compile_definitions(mscclpp_gpunetio_obj PRIVATE USE_IBVERBS)
    target_include_directories(mscclpp_gpunetio_obj SYSTEM PRIVATE ${IBVERBS_INCLUDE_DIRS})
endif()
if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_gpunetio_obj PRIVATE MSCCLPP_USE_CUDA)
endif()

set_target_properties(mscclpp_gpunetio_obj PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE 1)

target_link_libraries(mscclpp_gpunetio_obj PRIVATE ${GPU_LIBRARIES} dl)

install(DIRECTORY ${GPUNETIO_VENDOR_HOME}/include/ DESTINATION include/mscclpp/gpunetio)
install(FILES ${GPUNETIO_VENDOR_HOME}/LICENSE.txt DESTINATION share/licenses/mscclpp/gpunetio)
install(TARGETS mscclpp_gpunetio_headers EXPORT mscclppTargets)
