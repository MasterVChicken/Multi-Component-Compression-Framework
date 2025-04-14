cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(mgard)
find_package(zfp REQUIRED)
find_package(SZ3 REQUIRED)




option(ENABLE_CUDA "Enable CUDA compressors." OFF)




add_library(cpu-compressors INTERFACE)
target_include_directories(cpu-compressors INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cpu-compressors INTERFACE zfp::zfp SZ3::SZ3 mgard::mgard)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cpu-compressors INTERFACE OpenMP::OpenMP_CXX)
endif()


if(${ENABLE_CUDA})
    set(CMAKE_CUDA_ARCHITECTURES 90)

    include_directories(/home/leonli/cuSZp/install/include)

    enable_language(CUDA)
    find_package(cuSZp REQUIRED)
    add_compile_definitions(ENABLE_CUDA_COMPRESSOR=1)
    
    set(gpu_compressors_src
        "include/GPUMGARDCompressor.hpp"
        "include/GPUSZCompressor.hpp"
        "include/GPUZFPCompressor.hpp")
    
    set_source_files_properties(${gpu_compressors_src} PROPERTIES LANGUAGE CUDA)
    add_library(gpu-compressors ${gpu_compressors_src})
    target_link_libraries(gpu-compressors PUBLIC mgard::mgard cuSZp::cuSZp_shared zfp::zfp)
    set_target_properties(gpu-compressors PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    add_executable(ProgressiveCompression main.cpp)
    target_link_libraries(ProgressiveCompression gpu-compressors cpu-compressors)
else()
    add_compile_definitions(ENABLE_CUDA_COMPRESSOR=0)
    add_executable(ProgressiveCompression main.cpp)
    target_link_libraries(ProgressiveCompression cpu-compressors)
endif()

target_compile_options(ProgressiveCompression PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)
