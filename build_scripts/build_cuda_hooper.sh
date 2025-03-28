#!/bin/sh

set -e
set -x

#  Get root directory
project_root=$(pwd)

######## User Configurations ########
# Source directory
src_dir=${project_root}
# Build directory
build_dir=${project_root}/build-cuda-hooper
# Number of processors used for building
num_build_procs=$1
# Installtaion directory
install_dir=${project_root}/install-cuda-hooper
# CUDA architecture version
cuda_arch="90"

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export CUDACXX=nvcc

######################
# build MGARD
######################
mgard_dir=${build_dir}/mgard
mgard_src_dir=${mgard_dir}/src
mgard_build_dir=${mgard_dir}/build
mgard_install_dir=${install_dir}
if [ ! -d "${mgard_src_dir}" ]; then
  git clone -b 1.5.2 https://github.com/CODARcode/MGARD.git ${mgard_src_dir}
fi
cd ${mgard_src_dir}
# dynamically adjust cuda arch version to user configuration
sed "s/-DCMAKE_CUDA_ARCHITECTURES=\"80\"/-DCMAKE_CUDA_ARCHITECTURES=\"${cuda_arch}\"/" \
  build_scripts/build_mgard_cuda_ampere.sh | bash
cd ${project_root}

######################
# build ZFP
######################
zfp_dir=${build_dir}/zfp
zfp_src_dir=${zfp_dir}/src
zfp_install_dir=${install_dir}
if [ ! -d "${zfp_src_dir}" ]; then
  git clone -b 1.0.0 https://github.com/LLNL/zfp.git ${zfp_src_dir}
fi
cd ${zfp_src_dir}
mkdir -p build
cd build
# Use CMake to build ZFP and set installation directory
cmake .. -DCMAKE_INSTALL_PREFIX=${zfp_install_dir}
cmake --build . --config Release -j ${num_build_procs}
ctest --output-on-failure
cmake --install .
cd ${project_root}

######################
# build SZ
######################
sz_dir=${build_dir}/sz
sz_src_dir=${sz_dir}/src
sz_install_dir=${install_dir}
if [ ! -d "${sz_src_dir}" ]; then
  git clone -b v3.1.7 https://github.com/szcompressor/SZ3.git ${sz_src_dir}
fi
cd ${sz_src_dir}
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${sz_install_dir} ..
make -j ${num_build_procs}
make install

