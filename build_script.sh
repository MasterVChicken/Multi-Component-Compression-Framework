#!/bin/sh

set -x
set -e

export CC=gcc
export CXX=g++
export CUDACXX=nvcc

source_dir=`pwd`
external_dir=${source_dir}/external
mkdir -p external
sz3_dir=${external_dir}/SZ3
mgard_dir=${external_dir}/MGARD
zfp_cpu_dir=${external_dir}/zfp_cpu
zfp_gpu_dir=${external_dir}/zfp_gpu


# Building SZ3
cd ${external_dir}
if [ ! -d "${sz3_dir}" ]; then
  git clone https://github.com/szcompressor/SZ3.git
fi
cd SZ3
git reset --hard c49fd17f2d908835c41000c1286c510046c0480e
mkdir -p build
mkdir -p install
cd build
cmake -DCMAKE_INSTALL_PREFIX=${external_dir}/SZ3/install ..
make
make install


# Building MGARD
cd ${external_dir}
if [ ! -d "${mgard_dir}" ]; then
  git clone https://github.com/CODARcode/MGARD.git
fi
cd MGARD
git reset --hard 75cbf6cfa14f069b2d155b3267a59d3792506ff4
./build_scripts/build_mgard_cuda_ampere.sh 8


# Build ZFP CPU
cd ${external_dir}
if [ ! -d "${zfp_cpu_dir}" ]; then
    git clone https://github.com/LLNL/zfp.git zfp_cpu
fi
cd zfp_cpu
git reset --hard f40868a6a1c190c802e7d8b5987064f044bf7812
mkdir -p build
cd build
cmake -DBUILD_TESTING=OFF -DZFP_WITH_OPENMP=ON ..
make

# Build ZFP GPU
cd ${external_dir}
if [ ! -d "${zfp_gpu_dir}" ]; then
    git clone https://github.com/LLNL/zfp.git zfp_gpu
fi
cd zfp_gpu
git reset --hard f40868a6a1c190c802e7d8b5987064f044bf7812
mkdir -p build
cd build
cmake -DBUILD_TESTING=OFF -DZFP_WITH_CUDA=ON ..
make


mgard_install_dir=${external_dir}/MGARD/install-cuda-ampere
sz3_install_dir=${external_dir}/SZ3/install
zfp_cpu_build_dir=${external_dir}/zfp_cpu/build
zfp_gpu_build_dir=${external_dir}/zfp_gpu/build

export LD_LIBRARY_PATH=${mgard_install_dir}/lib:${mgard_install_dir}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${sz3_install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${zfp_cpu_build_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${zfp_gpu_build_dir}/lib:$LD_LIBRARY_PATH

cd ${source_dir}
rm -rf ./build

cmake -S . -B ./build \
        -DENABLE_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release

cmake --build ./build -j$(nproc)

echo "Build completed successfully!"