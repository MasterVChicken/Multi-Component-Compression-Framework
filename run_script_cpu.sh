#!/bin/bash
set -e
set -x

# OpenMP settings
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=true
export OMP_PLACES=cores

EXEC=./build/ProgressiveCompressionCPU

# ---- Test1: ${IN_DATA} ----
IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
ndims=3
dims=(512 512 512)
precision="s"
errorCount=6
errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)

echo "--- [${CUDA_FLAG}] Test1: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test1 ---"

# ---- Test 2 ----
IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500/Pf48.bin.f32
ndims=3
dims=(500 500 100)
precision="s"
errorCount=6
errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)

echo "--- Test2: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test2 ---"

# ---- Test 3 ----
IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-SCALE_98x1200x1200/PRES-98x1200x1200.f32
ndims=3
dims=(1200 1200 98)
precision="s"
errorCount=6
errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)

echo "Test3: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test3 ---"

# ---- Test 4 ----
IN_DATA=/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/velocityz.d64
ndims=3
dims=(384 384 256)
precision="d"
errorCount=6
errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)

echo "Test4: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test4 ---"

