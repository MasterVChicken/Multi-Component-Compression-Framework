cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 90)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(SRC_FILES
    src/main.cu
)

add_executable(ProgressiveCompression ${SRC_FILES})

set_target_properties(ProgressiveCompression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(mgard)
find_package(zfp REQUIRED)
# find_package(CUSZ REQUIRED)

target_link_libraries(ProgressiveCompression PUBLIC mgard::mgard zfp::zfp)




