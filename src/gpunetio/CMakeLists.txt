# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Vendored DOCA GPUNetIO (GDAKI) sources, imported from the NCCL GIN gdaki
# backend (BSD-3-Clause). Provides GPU-initiated networking (kernel-issued
# RDMA WRITE/atomic) used by the GPUNetIO PortChannel backend.
#
# Built as an OBJECT library with relaxed warnings (third-party code) and
# linked into mscclpp_obj. Talks to the NIC purely via dlopen of
# libibverbs/libmlx5/libcuda, so it adds no new NEEDED shared libraries.

set(GPUNETIO_HOME ${CMAKE_CURRENT_SOURCE_DIR})

set(GPUNETIO_SOURCES
    ${GPUNETIO_HOME}/src/doca_verbs_qp.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_cq.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_device_attr.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_umem.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_srq.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_uar.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_log.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_high_level.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_cuda_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_mlx5dv_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_verbs_ibv_wrapper.cpp
    ${GPUNETIO_HOME}/src/doca_gpunetio_gdrcopy.cpp
)

# MSCCL++ host service that drives the vendored DOCA host API to bring up the
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

# Vendored headers: expose include/ and include/host to the DOCA sources.
target_include_directories(mscclpp_gpunetio_obj PUBLIC
    ${GPUNETIO_HOME}/include
    ${GPUNETIO_HOME}/include/host)

target_include_directories(mscclpp_gpunetio_obj PRIVATE
    ${GPUNETIO_HOME}/src
    ${GPUNETIO_HOME}
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include)

# Third-party code: do not enforce mscclpp's -Wall -Wextra -Werror.
target_compile_options(mscclpp_gpunetio_obj PRIVATE -w)

# DOCA_VERBS_USE_NET_WRAPPER selects the dlopen-based ibverbs/mlx5dv wrappers, so
# the vendored code adds no new NEEDED shared libraries (mirrors NCCL's build).
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
