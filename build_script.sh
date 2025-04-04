#!/bin/sh

set -x
set -e

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export CUDACXX=nvcc

mgard_install_dir=/home/leonli/MGARD/install-cuda-ampere
zfp_install_dir=/home/leonli/zfp-install

mkdir -p build 
cmake -S .  -B ./build \
        -DCMAKE_PREFIX_PATH="${mgard_install_dir};${zfp_install_dir}"
cmake --build ./build
