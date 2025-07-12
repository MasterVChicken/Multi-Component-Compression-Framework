#!/bin/sh

set -x
set -e

export CC=gcc
export CXX=g++
export CUDACXX=nvcc
export LD=/home/leonli/miniconda/envs/pytorch_build_env/bin/x86_64-conda-linux-gnu-ld

ENABLE_CUDA=${1:-OFF}


# GPU compressor paths
mgard_install_dir=/home/leonli/MGARD/install-cuda-hooper
zfp_install_dir_gpu=/home/leonli/zfp-gpu

# CPU compressor paths
zfp_install_dir_cpu=/home/leonli/zfp-cpu
sz3_install_dir=/home/leonli/sz3-install

export LD_LIBRARY_PATH=${mgard_install_dir}/lib:${mgard_install_dir}/lib64:$LD_LIBRARY_PATH

mkdir -p build 
cmake -S .  -B ./build \
        -DENABLE_CUDA=${ENABLE_CUDA}\
        -DCMAKE_LINKER=$LD\
        -DCMAKE_PREFIX_PATH="${mgard_install_dir};${zfp_install_dir_gpu};${zfp_install_dir_cpu};${sz3_install_dir}"
cmake --build ./build
