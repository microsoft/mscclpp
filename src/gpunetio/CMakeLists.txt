# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DOCA GPUNetIO (GDAKI) sources from the pinned NVIDIA-DOCA/gpunetio submodule.
# Provides GPU-initiated networking (kernel-issued RDMA WRITE/atomic) used by
# the GPUNetIO PortChannel backend.
#
# Built as an OBJECT library with relaxed warnings (third-party code) and
# linked into mscclpp_obj.

set(GPUNETIO_HOME ${CMAKE_CURRENT_SOURCE_DIR})
set(GPUNETIO_VENDOR_HOME ${GPUNETIO_HOME}/vendor)

message(STATUS "GPUNetIO: direct GPU doorbells and GPU-resident CQs required; CPU-assisted fallback disabled")

if(NOT EXISTS ${GPUNETIO_VENDOR_HOME}/include/doca_gpunetio_host.h)
    message(FATAL_ERROR
        "GPUNetIO submodule is not initialized. Run: git submodule update --init --recursive")
endif()

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
    ${GPUNETIO_VENDOR_HOME}/src/doca_gpunetio_high_level.cpp
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

# Expose upstream and MSCCL++ service headers.
target_include_directories(mscclpp_gpunetio_obj PUBLIC
    ${GPUNETIO_VENDOR_HOME}/include
    ${GPUNETIO_VENDOR_HOME}/include/host)

target_include_directories(mscclpp_gpunetio_obj PRIVATE
    ${GPUNETIO_VENDOR_HOME}/src
    ${GPUNETIO_HOME}
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include)

# Third-party code: do not enforce mscclpp's -Wall -Wextra -Werror.
target_compile_options(mscclpp_gpunetio_obj PRIVATE -w)

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
