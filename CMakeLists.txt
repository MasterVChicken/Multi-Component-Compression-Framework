cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MGARD_ROOT "${PROJECT_SOURCE_DIR}/build-cuda-hooper/mgard/src/install-cuda-ampere")



include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MGARD_ROOT}/include)

link_directories(${MGARD_ROOT}/lib ${MGARD_ROOT}/lib64)


add_executable(ProgressiveCompression main.cpp)


target_link_libraries(ProgressiveCompression mgard)
