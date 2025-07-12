#!/bin/bash
set -e
set -x

EXEC=./build/ProgressiveCompressionGPUMGARD

# ---- Test1: ${IN_DATA} ----
IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
ndims=3
dims=(512 512 512)
precision="s"
errorCount=6
# errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
# The following preset relative errorbound corresponding to actual error relative bound ahead
errorList=(1.01e+1 6.4e-1 6e-2 7e-3 5.2e-4 6e-5)

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
# errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
# The following preset relative errorbound corresponding to actual error relative bound ahead
errorList=(3.97e+0 3.95e-1 4e-2 4.2e-3 4.1e-4 4e-5)

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
# errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
# The following preset relative errorbound corresponding to actual error relative bound ahead
errorList=(5.97e+0 4.12e-1 4e-2 4.2e-3 3.9e-4 4e-5)

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
# errorList=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
# The following preset relative errorbound corresponding to actual error relative bound ahead
errorList=(5.57e+0 5.8e-1 6.8e-2 4.5e-3 4.9e-4 5e-5)

echo "Test4: ${IN_DATA} ---"
"${EXEC}" \
  -i "${IN_DATA}" \
  -n "${ndims}" "${dims[@]}" \
  -ECount "${errorCount}" \
  -E "${errorCount}" "${errorList[@]}" \
  -p "${precision}"
echo "--- Done Test4 ---"

