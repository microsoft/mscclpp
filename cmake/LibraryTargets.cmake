# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Library targets configuration

function(configure_mscclpp_object_library)
    add_library(mscclpp_obj OBJECT)

    # Include directories
    target_include_directories(mscclpp_obj SYSTEM PRIVATE
        ${GPU_INCLUDE_DIRS}
        ${NUMA_INCLUDE_DIRS})

    # Link libraries
    target_link_libraries(mscclpp_obj PRIVATE
        ${GPU_LIBRARIES}
        ${NUMA_LIBRARIES}
        nlohmann_json::nlohmann_json
        Threads::Threads
        dl)

    # IBVerbs support if available
    if(IBVERBS_FOUND)
        target_include_directories(mscclpp_obj SYSTEM PRIVATE ${IBVERBS_INCLUDE_DIRS})
        target_link_libraries(mscclpp_obj PRIVATE ${IBVERBS_LIBRARIES})
        target_compile_definitions(mscclpp_obj PUBLIC USE_IBVERBS)
    endif()

    # Target properties
    set_target_properties(mscclpp_obj PROPERTIES
        LINKER_LANGUAGE CXX
        POSITION_INDEPENDENT_CODE ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${MSCCLPP_SOVERSION})

    # Compile definitions
    if(MSCCLPP_USE_CUDA)
        target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_CUDA)
    elseif(MSCCLPP_USE_ROCM)
        target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_USE_ROCM)
    endif()

    if(MSCCLPP_ENABLE_TRACE)
        target_compile_definitions(mscclpp_obj PRIVATE MSCCLPP_ENABLE_TRACE)
    endif()

    if(MSCCLPP_NPKIT_FLAGS)
        target_compile_definitions(mscclpp_obj PRIVATE ${MSCCLPP_NPKIT_FLAGS})
    endif()
endfunction()

function(create_mscclpp_libraries)
    # Shared library
    add_library(mscclpp SHARED)
    target_link_libraries(mscclpp PUBLIC mscclpp_obj)
    set_target_properties(mscclpp PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${MSCCLPP_SOVERSION})

    # Static library
    add_library(mscclpp_static STATIC)
    target_link_libraries(mscclpp_static PUBLIC mscclpp_obj)
    set_target_properties(mscclpp_static PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${MSCCLPP_SOVERSION})
endfunction()

# Configure the main library
configure_mscclpp_object_library()
create_mscclpp_libraries()
