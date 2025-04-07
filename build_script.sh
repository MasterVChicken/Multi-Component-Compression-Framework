#!/bin/sh

set -x
set -e

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export CUDACXX=nvcc

# GPU compressor paths
mgard_install_dir=/home/leonli/MGARD/install-cuda-ampere
cuszp_install_dir=/home/leonli/cuSZp/install
zfp_install_dir_gpu=/home/leonli/zfp-gpu

# CPU compressor paths
zfp_install_dir_cpu=/home/leonli/zfp-cpu
sz3_install_dir=/home/leonli/sz3-install

mkdir -p build 
cmake -S .  -B ./build \
        -DCMAKE_PREFIX_PATH="${mgard_install_dir};${cuszp_install_dir};${zfp_install_dir_gpu};${zfp_install_dir_cpu};${sz3_install_dir}"
cmake --build ./build
