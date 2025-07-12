cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(mgard)
find_package(zfp REQUIRED)
find_package(SZ3 REQUIRED)

# Compile CPU Compressors
option(ENABLE_CUDA "Enable CUDA compressors." ON)

add_library(cpu-compressors INTERFACE)
target_include_directories(cpu-compressors INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cpu-compressors INTERFACE zfp::zfp SZ3::SZ3 mgard::mgard)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cpu-compressors INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(ProgressiveCompressionCPU testCPUCompressors.cpp)
target_link_libraries(ProgressiveCompressionCPU cpu-compressors)

target_compile_options(ProgressiveCompressionCPU PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# -------------------------------
# GPU Compressors + Executables
# -------------------------------
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Compressors only (no test .cpp)
set(gpu_compressors_src
    include/GPUMGARDCompressor.hpp
    include/GPUZFPCompressor.hpp
)
set_source_files_properties(${gpu_compressors_src} PROPERTIES LANGUAGE CUDA)

add_library(gpu-compressors ${gpu_compressors_src})
target_link_libraries(gpu-compressors PUBLIC mgard::mgard zfp::zfp)
target_include_directories(gpu-compressors PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(gpu-compressors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)


# -------------------------------
# GPU Executables (test .cpp files)
# -------------------------------
# MGARD
set_source_files_properties(testMGARDCompressor.cpp PROPERTIES LANGUAGE CUDA)
add_executable(ProgressiveCompressionGPUMGARD testMGARDCompressor.cpp)
target_link_libraries(ProgressiveCompressionGPUMGARD gpu-compressors)
target_include_directories(ProgressiveCompressionGPUMGARD PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(gpu-compressors PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)


# ZFP
set_source_files_properties(testZFPCompressor.cpp PROPERTIES LANGUAGE CUDA)
add_executable(ProgressiveCompressionGPUZFP testZFPCompressor.cpp)
target_link_libraries(ProgressiveCompressionGPUZFP gpu-compressors)
target_include_directories(ProgressiveCompressionGPUZFP PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(gpu-compressors PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)
