cmake_minimum_required(VERSION 3.16)

project(fast_topk VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(ENABLE_AVX2 "Enable AVX2 support" ON)

# Add include directories for the library (header-only, so no source files to compile)
add_library(fast_topk INTERFACE)

target_include_directories(fast_topk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_AVX2)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        check_cxx_compiler_flag(/arch:AVX2 AVX2_SUPPORTED)
    else()
        check_cxx_compiler_flag(-mavx2 AVX2_SUPPORTED)
    endif()

    if (AVX2_SUPPORTED)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(fast_topk INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        else()
            target_compile_options(fast_topk INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
        endif()
    endif()
endif ()
