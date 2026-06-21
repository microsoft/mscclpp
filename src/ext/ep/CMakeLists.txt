# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Builds `mscclpp_ep_cpp`, a pybind11 + PyTorch C++ extension that exposes the
# EP (Mixture-of-Experts dispatch/combine) Buffer to Python. This module is
# separate from the nanobind-based `_mscclpp` because EP carries `torch::Tensor`
# through its ABI (ported verbatim from DeepEP).
#
# Requires: PyTorch with its CMake integration available on CMAKE_PREFIX_PATH.
# Easiest invocation:
#     cmake -S . -B build \
#           -DMSCCLPP_BUILD_EXT_EP=ON \
#           -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(WARNING
        "MSCCLPP_BUILD_EXT_EP=ON but PyTorch CMake package was not found. "
        "Set CMAKE_PREFIX_PATH to `$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')` "
        "or disable MSCCLPP_BUILD_EXT_EP. Skipping the EP extension.")
    return()
endif()

find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    # PyTorch ships a bundled pybind11 we can fall back on.
    get_target_property(_torch_include_dirs torch INTERFACE_INCLUDE_DIRECTORIES)
    foreach(d ${_torch_include_dirs})
        if(EXISTS "${d}/pybind11/pybind11.h")
            set(PYBIND11_INCLUDE_DIR "${d}")
            break()
        endif()
    endforeach()
    if(NOT PYBIND11_INCLUDE_DIR)
        message(FATAL_ERROR
            "pybind11 not found and not bundled with Torch. Install it (pip install pybind11) "
            "or provide -Dpybind11_DIR=...")
    endif()
endif()

file(GLOB_RECURSE EP_SOURCES CONFIGURE_DEPENDS
    buffer.cc
    bindings.cpp
    kernels/*.cu
)

if(MSCCLPP_USE_ROCM)
    # ROCm port of the EP kernels is not supported yet; see src/ext/ep/README.md.
    message(WARNING "mscclpp_ep: ROCm build path not implemented, falling back to CXX compile.")
    file(GLOB_RECURSE EP_CU_SOURCES kernels/*.cu)
    set_source_files_properties(${EP_CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

# Build as a Python extension module (shared object with Python ABI suffix).
# The name `mscclpp_ep_cpp` matches the `TORCH_EXTENSION_NAME` hard-coded in
# `bindings.cpp` / `buffer.hpp`.
Python_add_library(mscclpp_ep_cpp MODULE ${EP_SOURCES})

target_compile_definitions(mscclpp_ep_cpp PRIVATE TORCH_EXTENSION_NAME=mscclpp_ep_cpp)
target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_AVOID_BFLOAT16_ALIAS)
target_include_directories(mscclpp_ep_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/core/include
    ${PROJECT_SOURCE_DIR}/src/ext/include
    ${GPU_INCLUDE_DIRS}
)
if(pybind11_FOUND)
    target_link_libraries(mscclpp_ep_cpp PRIVATE pybind11::module)
else()
    target_include_directories(mscclpp_ep_cpp SYSTEM PRIVATE ${PYBIND11_INCLUDE_DIR})
endif()

target_link_libraries(mscclpp_ep_cpp PRIVATE ${TORCH_LIBRARIES})
# Torch's CUDA interop library (ATen CUDAStream helpers used in buffer.cc).
if(TARGET torch::torch)
    target_link_libraries(mscclpp_ep_cpp PRIVATE torch::torch)
endif()
# libtorch_python contains pybind11 bindings for torch::Tensor / torch::dtype
# (symbols like THPDtypeType). It is not listed in TORCH_LIBRARIES.
find_library(TORCH_PYTHON_LIBRARY torch_python
    HINTS "${TORCH_INSTALL_PREFIX}/lib")
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(mscclpp_ep_cpp PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
    message(WARNING "libtorch_python not found; `import mscclpp_ep_cpp` may fail at runtime.")
endif()
target_link_libraries(mscclpp_ep_cpp PRIVATE mscclpp ${GPU_LIBRARIES} Threads::Threads)

# Number of intra-node NVLink peers compiled into the EP kernels.
#  - 8 (default) matches DeepEP upstream and H100/HGX 8-GPU nodes.
#  - 4 is required for Azure GB200 NVL72 (4 GPUs per NUMA host).
set(MSCCLPP_EP_NUM_MAX_NVL_PEERS "8" CACHE STRING
    "Compile-time NUM_MAX_NVL_PEERS for the EP kernels (8 for HGX, 4 for GB200)")
target_compile_definitions(mscclpp_ep_cpp PRIVATE
    NUM_MAX_NVL_PEERS=${MSCCLPP_EP_NUM_MAX_NVL_PEERS})

# Kernel-side debug timeout (~10s) — set via:
#   -DMSCCLPP_EP_KERNEL_DEBUG_TIMEOUT=ON
option(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT
       "Use a short ~10s kernel spin timeout (default is ~100s)" OFF)
if(MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_EP_KERNEL_DEBUG_TIMEOUT)
endif()

# Experimental: NCCL-EP-style warp-specialized HT dispatch path.
#  When ON, the internode `dispatch` kernel selects an alternate code path
#  (guarded by `EP_DISPATCH_NCCLEP`) that ports NVIDIA NCCL-EP's
#  warp-specialized overlap pipeline (concurrent inter-node fabric-write +
#  intra-node drain) onto mscclpp MemoryChannel put/signal primitives, to
#  close the GB200 MNNVL dispatch gap vs NCCL-EP. Default OFF = production path.
#   -DMSCCLPP_EP_DISPATCH_NCCLEP=ON
option(MSCCLPP_EP_DISPATCH_NCCLEP
       "Use the experimental NCCL-EP-ported warp-specialized HT dispatch kernel" OFF)
if(MSCCLPP_EP_DISPATCH_NCCLEP)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE EP_DISPATCH_NCCLEP)
endif()

set_target_properties(mscclpp_ep_cpp PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET default
)

# Install layout.
#  - scikit-build / wheel build (SKBUILD set by scikit-build-core):
#       module lands next to the `mscclpp` python package; libmscclpp.so is
#       under `mscclpp/lib/`, so rpath = `$ORIGIN/mscclpp/lib`.
#  - Plain CMake install: standard `${INSTALL_PREFIX}/lib` with rpath
#       `$ORIGIN/../lib` so the .so finds the sibling mscclpp shared lib.
if(DEFINED SKBUILD OR DEFINED ENV{SKBUILD})
    set_target_properties(mscclpp_ep_cpp PROPERTIES
        INSTALL_RPATH "\$ORIGIN/mscclpp/lib")
    install(TARGETS mscclpp_ep_cpp LIBRARY DESTINATION ..)
else()
    set_target_properties(mscclpp_ep_cpp PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib")
    install(TARGETS mscclpp_ep_cpp
        LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
endif()

if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_ep_cpp PRIVATE MSCCLPP_USE_ROCM)
endif()
