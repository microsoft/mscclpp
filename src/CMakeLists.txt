file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.h)
file(GLOB to_remove gdr.cc)
list(REMOVE_ITEM SOURCES ${to_remove})

target_sources(mscclpp PRIVATE ${SOURCES})
