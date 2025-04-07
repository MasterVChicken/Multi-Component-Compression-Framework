cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 90)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Maybe bugs for cuSZp?
include_directories(/home/leonli/cuSZp/install/include)

set(SRC_FILES
    src/main.cu
)

add_executable(ProgressiveCompression ${SRC_FILES})

set_target_properties(ProgressiveCompression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(mgard)
find_package(zfp REQUIRED)
find_package(cuSZp REQUIRED)
find_package(SZ3 REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ProgressiveCompression PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(ProgressiveCompression PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

target_link_libraries(ProgressiveCompression PUBLIC mgard::mgard zfp::zfp cuSZp::cuSZp_shared SZ3::SZ3)


