add_subdirectory(collectives)

if(MSCCLPP_BUILD_APPS_NCCL)
    add_subdirectory(nccl)
endif()
