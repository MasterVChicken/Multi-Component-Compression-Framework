#!/bin/bash
set -e
set -x

EXEC=./build/ProgressiveCompressionGPUZFP
# All errorList here containthe actual fixed rate correponding to REL tolerance of 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6

# ---- Test1: ${IN_DATA} ----
IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
ndims=3
dims=(512 512 512)
precision="s"
errorCount=6
# errorList=(10 9 8 7 6 5)
errorList=(4.15 5 5 5.6 5 4.4)

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
errorList=(5.8 5.5 5 4.5 5.3 5.1)

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
errorList=(1.5 5.2 5.5 4.8 5.3 5.3)

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
errorList=(5.5 5 5.5 5 5 5.5)

echo "Test4: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test4 ---"

