# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

nanobind_add_module(mscclpp_py core_py.cpp)
set_target_properties(mscclpp_py PROPERTIES OUTPUT_NAME mscclpp)
target_link_libraries(mscclpp_py PRIVATE mscclppStatic)
if (SKBUILD)
    install(TARGETS mscclpp_py LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
else()
    install(TARGETS mscclpp_py LIBRARY DESTINATION lib)
endif()
