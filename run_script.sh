#!/bin/bash
set -e
set -x

# compile executables
./build_script.sh 32

# executable file path
exec=./build/ProgressiveCompression


export OMP_NUM_THREADS=64
export OMP_PROC_BIND=true
export OMP_PLACES=cores

IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32

# denote dim of datasets and dims
ndims=3
dims=(512 512 512)  

# set num of tolerances and toleranceList(ABS)
errorCount=11
errorList="478030.252417 239015.1262085 47803.0252417 23901.51262085 4780.30252417 2390.151262085 478.030252417 239.0151262085 47.8030252417 23.9015126208 4.7803025242"

# s represents float and d represents double
precision="s"

echo "Starting Test dataset: $IN_DATA"
$exec -i $IN_DATA -n $ndims "${dims[@]}" -ECount $errorCount -E $errorCount $errorList -p $precision

echo "Ending Test for dataset: $IN_DATA"






IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500/Pf48.bin.f32

ndims=3
dims=(500 500 100)  

errorCount=11
errorList="341.1740723 170.58703615 34.11740723 17.058703615 3.411740723 1.7058703615 0.3411740723 0.17058703615 0.03411740723 0.017058703615 0.003411740723"

precision="s"

echo "Starting Test dataset: $IN_DATA"
$exec -i $IN_DATA -n $ndims "${dims[@]}" -ECount $errorCount -E $errorCount $errorList -p $precision

echo "Ending Test for dataset: $IN_DATA"








IN_DATA=/home/leonli/SDRBENCH/single_precision/SDRBENCH-SCALE_98x1200x1200/PRES-98x1200x1200.f32

ndims=3
dims=(1200 1200 98)  

errorCount=11
errorList="10182.021875 5091.0109375 1018.2021875 509.10109375 101.82021875 50.910109375 10.182021875 5.0910109375 1.0182021875 0.50910109375 0.10182021875"

precision="s"

echo "Starting Test dataset: $IN_DATA"
$exec -i $IN_DATA -n $ndims "${dims[@]}" -ECount $errorCount -E $errorCount $errorList -p $precision

echo "Ending Test for dataset: $IN_DATA"









IN_DATA=/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/velocityz.d64

ndims=3
dims=(384 384 256)  

errorCount=11
errorList="0.899611000000 0.449805500000 0.089961100000 0.044980550000 0.008996110000 0.004498055000 0.000899611000 0.000449805500 0.000089961100 0.000044980550 0.000008996110"

precision="d"

echo "Starting Test dataset: $IN_DATA"
$exec -i $IN_DATA -n $ndims "${dims[@]}" -ECount $errorCount -E $errorCount $errorList -p $precision

echo "Ending Test for dataset: $IN_DATA"
