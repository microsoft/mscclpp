project(mscclpp)
cmake_minimum_required(VERSION 3.18...3.22)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Create CMake targets for all Python components needed by nanobind
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
else()
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

set(CUDA_DIR "/usr/local/cuda")

set(MSCCLPP_DIR ${CMAKE_CURRENT_LIST_DIR}/../build)

nanobind_add_module(
  _py_mscclpp
  NOSTRIP
  NB_STATIC
  src/_py_mscclpp.cpp
)

target_include_directories(
  _py_mscclpp
  PUBLIC
  ${CUDA_DIR}/include
  ${MSCCLPP_DIR}/include
)
target_link_directories(
  _py_mscclpp
  PUBLIC
  ${CUDA_DIR}/lib
  ${MSCCLPP_DIR}/lib
)

target_link_libraries(
  _py_mscclpp
  PUBLIC
  mscclpp
)

add_custom_target(build-package ALL DEPENDS _py_mscclpp)
add_custom_command(
  TARGET build-package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mscclpp
  ${CMAKE_CURRENT_BINARY_DIR}/mscclpp)

add_custom_command(
  TARGET build-package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  $<TARGET_FILE:_py_mscclpp>
  ${CMAKE_CURRENT_BINARY_DIR}/mscclpp/)

add_custom_command(
  TARGET build-package POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${MSCCLPP_DIR}/lib/libmscclpp.so
  ${CMAKE_CURRENT_BINARY_DIR}/mscclpp/)

