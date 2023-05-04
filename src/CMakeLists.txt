file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.h)
file(GLOB to_remove gdr.cc)
list(REMOVE_ITEM SOURCES ${to_remove})

add_library(mscclpp SHARED ${SOURCES})
set_target_properties(mscclpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(mscclpp PRIVATE MSCCLPP::ibverbs MSCCLPP::numa CUDA::cudart CUDA::cuda_driver)
if(GDRCOPY_FOUND)
    target_link_libraries(mscclpp PRIVATE MSCCLPP::gdrcopy)
endif()

target_include_directories(mscclpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)