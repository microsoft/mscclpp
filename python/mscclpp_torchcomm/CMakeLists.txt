# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

include(FetchContent)

# Fetch torchcomms headers (header-only dependency — we only need the interface headers)
FetchContent_Declare(torchcomms
    GIT_REPOSITORY https://github.com/meta-pytorch/torchcomms.git
    GIT_TAG v0.2.0-rc2
)
FetchContent_GetProperties(torchcomms)
if(NOT torchcomms_POPULATED)
    FetchContent_Populate(torchcomms)
endif()

# Find PyTorch (provides Torch libraries and Python development headers)
find_package(Torch REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11 via Python package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FIND_RESULT
)
if(PYBIND11_FIND_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 REQUIRED)

# Gather our C++ sources
file(GLOB_RECURSE TORCHCOMM_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

# Torchcomms framework sources we need to compile in directly.
# Our module inherits from TorchWork, TorchCommBackend, and registers with
# TorchCommFactory — these symbols must be in our .so since torchcomms doesn't
# export them from a shared lib we can link against.
set(TORCHCOMMS_FRAMEWORK_SOURCES
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/TorchWork.cpp
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/TorchCommFactory.cpp
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/TorchCommOptions.cpp
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/TorchCommTypes.cpp
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/utils/Utils.cpp
    ${torchcomms_SOURCE_DIR}/comms/torchcomms/utils/StoreManager.cpp
)

# MSCCL++ algorithm selector (same one used by the NCCL extension)
set(MSCCLPP_ALGO_SELECTOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ext/nccl/algorithm_selector.cc
)

# Build pybind11 module
pybind11_add_module(_comms_mscclpp ${TORCHCOMM_SOURCES} ${TORCHCOMMS_FRAMEWORK_SOURCES} ${MSCCLPP_ALGO_SELECTOR_SOURCES})

# Find glog (required by torchcomms framework sources via Logging.hpp).
# Derive the conda env prefix from the Python executable path so we can
# locate glog headers and libraries installed in the same environment.
get_filename_component(CONDA_PREFIX "${Python_EXECUTABLE}" DIRECTORY)
get_filename_component(CONDA_PREFIX "${CONDA_PREFIX}" DIRECTORY)
find_library(GLOG_LIBRARY glog HINTS "${CONDA_PREFIX}/lib")
find_path(GLOG_INCLUDE_DIR glog/logging.h HINTS "${CONDA_PREFIX}/include")

target_include_directories(_comms_mscclpp SYSTEM PRIVATE
    # torchcomms headers: resolves #include <comms/torchcomms/...>
    ${torchcomms_SOURCE_DIR}
    ${GPU_INCLUDE_DIRS}
)
# MSCCL++ internal headers (for algorithm_selector.hpp and debug.h)
target_include_directories(_comms_mscclpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ext/nccl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/core/include
)
if(GLOG_INCLUDE_DIR)
    target_include_directories(_comms_mscclpp SYSTEM PRIVATE ${GLOG_INCLUDE_DIR})
endif()

target_link_libraries(_comms_mscclpp PRIVATE
    # MUST use the shared library (not mscclpp_static) to avoid dual-singleton:
    # mscclpp_collectives.so links against libmscclpp.so, so if we statically
    # link mscclpp into our module, there are two copies of singletons like
    # UnixSocketServer::instance(). The bootstrap starts server #1 (static),
    # but the collectives code registers fds in server #2 (shared), causing
    # "Requested fd not found, size of fdSet_ is 0" crashes.
    mscclpp
    mscclpp_collectives
    ${TORCH_LIBRARIES}
    ${GPU_LIBRARIES}
)
if(GLOG_LIBRARY)
    target_link_libraries(_comms_mscclpp PRIVATE ${GLOG_LIBRARY})
endif()

# Propagate USE_ROCM define for mscclpp/gpu.hpp portability
target_compile_definitions(_comms_mscclpp PRIVATE
    $<$<BOOL:${MSCCLPP_USE_ROCM}>:USE_ROCM>
)

target_compile_features(_comms_mscclpp PRIVATE cxx_std_17)

# Set the torch_python library path for linking
set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
if(EXISTS "${TORCH_PYTHON_LIB}")
    target_link_libraries(_comms_mscclpp PRIVATE "${TORCH_PYTHON_LIB}")
endif()

# Copy built module to source tree for easy import
add_custom_target(torchcomm_lib_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_comms_mscclpp*.so
        ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS _comms_mscclpp
)
