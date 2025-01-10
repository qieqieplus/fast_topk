cmake_minimum_required(VERSION 3.16)

project(fast_topk_tests VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(ENABLE_AVX2 "Enable AVX2 support" ON)

include(FetchContent)
cmake_policy(SET CMP0135 NEW)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(benchmark)

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(bench benchmark)

add_executable(test test.cpp)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(test gtest_main)

include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(bench PRIVATE /O2)
    check_cxx_compiler_flag(/arch:AVX2 AVX2_SUPPORTED)
    if (ENABLE_AVX2 AND AVX2_SUPPORTED)
        add_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        add_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
else()
    target_compile_options(bench PRIVATE -O3)
    check_cxx_compiler_flag(-mavx2 AVX2_SUPPORTED)
    if (ENABLE_AVX2 AND AVX2_SUPPORTED)
        target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
        target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mfma>)

        target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
        target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    endif()
endif()
