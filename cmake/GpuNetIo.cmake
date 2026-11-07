# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Fetches and builds the third-party DOCA GPUNetIO (GDAKI) sources used by the GPU-initiated IB endpoint mode
# (EndpointConfig::Ib::Mode::Device). See docs/guide/gpunetio.md.
#
# Defines:
#   mscclpp_gpunetio_headers  INTERFACE target with the GPUNetIO include directories.
#   mscclpp_gpunetio_obj      OBJECT library with the GPUNetIO host sources, built with relaxed warnings.

include(FetchContent)
FetchContent_Declare(gpunetio
    GIT_REPOSITORY https://github.com/NVIDIA-DOCA/gpunetio.git
    GIT_TAG c5d97fd52346763bcfeaff593dd1d84215156218
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(gpunetio)
set(GPUNETIO_HOME ${gpunetio_SOURCE_DIR})

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
    ${GPUNETIO_HOME}/src/doca_gpunetio_high_level.cpp
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

install(DIRECTORY ${GPUNETIO_HOME}/include/ DESTINATION include)
install(FILES ${GPUNETIO_HOME}/LICENSE.txt DESTINATION share/licenses/mscclpp/gpunetio)
