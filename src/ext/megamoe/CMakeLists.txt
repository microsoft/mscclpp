# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(NOT MSCCLPP_USE_CUDA OR MSCCLPP_USE_ROCM)
    message(FATAL_ERROR "Native MegaMoE requires NVIDIA CUDA; ROCm is not supported.")
endif()
if(CUDAToolkit_VERSION VERSION_LESS "12.8")
    message(FATAL_ERROR "Native MegaMoE requires CUDA 12.8 or newer for SM100a.")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.3")
    message(FATAL_ERROR "Native MegaMoE requires CUDA nvcc/ptxas >=13.3 for packed FP8-to-BF16 conversion.")
endif()

set(MSCCLPP_MEGAMOE_CUTLASS_ROOT "" CACHE PATH "Standalone CUTLASS source directory for native MegaMoE")
if(NOT MSCCLPP_MEGAMOE_CUTLASS_ROOT)
    include(FetchContent)
    FetchContent_Declare(megamoe_cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG 147295a3d4b75f3aeff247c25b8927cea9a7006a
    )
    FetchContent_GetProperties(megamoe_cutlass)
    if(NOT megamoe_cutlass_POPULATED)
        FetchContent_Populate(megamoe_cutlass)
    endif()
    set(MSCCLPP_MEGAMOE_CUTLASS_ROOT "${megamoe_cutlass_SOURCE_DIR}")
endif()
if(NOT EXISTS "${MSCCLPP_MEGAMOE_CUTLASS_ROOT}/include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp")
    message(FATAL_ERROR "MegaMoE needs CUTLASS with the SM100 mixed-input collective. Set MSCCLPP_MEGAMOE_CUTLASS_ROOT.")
endif()

add_library(mscclpp_megamoe SHARED runtime.cc megamoe.cu)
target_include_directories(mscclpp_megamoe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
target_include_directories(mscclpp_megamoe SYSTEM PRIVATE "${MSCCLPP_MEGAMOE_CUTLASS_ROOT}/include")
target_compile_options(mscclpp_megamoe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>
)
target_link_libraries(mscclpp_megamoe PUBLIC mscclpp PRIVATE ${GPU_LIBRARIES})
target_compile_definitions(mscclpp_megamoe PRIVATE MSCCLPP_USE_CUDA)
set_target_properties(mscclpp_megamoe PROPERTIES
    CUDA_ARCHITECTURES "100a-real"
    POSITION_INDEPENDENT_CODE ON
    VERSION ${MSCCLPP_VERSION}
    SOVERSION ${MSCCLPP_SOVERSION}
    INSTALL_RPATH "$ORIGIN"
)
install(TARGETS mscclpp_megamoe LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
