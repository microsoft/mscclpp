cmake_minimum_required(VERSION 3.25)
project(mscclpp_consumer_check LANGUAGES CXX CUDA)

find_package(mscclpp CONFIG REQUIRED)
foreach(kind IN ITEMS shared static)
    add_executable(consumer_${kind} ../compile_tests.cu)
    target_compile_definitions(consumer_${kind} PRIVATE TEST_GPUNETIO_CONSUMER
        TEST_EXPECT_GPUNETIO=$<BOOL:${EXPECT_GPUNETIO}>)
    if(kind STREQUAL "shared")
        target_link_libraries(consumer_${kind} PRIVATE mscclpp::mscclpp)
    else()
        target_link_libraries(consumer_${kind} PRIVATE mscclpp::mscclpp_static)
    endif()
endforeach()