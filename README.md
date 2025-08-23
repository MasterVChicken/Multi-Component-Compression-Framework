# Multi-Component-Compression-Framework

GPU Compressor: 
- MGARD(75cbf6cfa14f069b2d155b3267a59d3792506ff4)
- ZFP(f40868a6a1c190c802e7d8b5987064f044bf7812)

CPU Compressor: 
- ZFP(f40868a6a1c190c802e7d8b5987064f044bf7812) 
- SZ3(c49fd17f2d908835c41000c1286c510046c0480e)

For compiling safety, please use GCC 8.5 - 11.x.

Make sure you adjust configuration and data path first.
For MGARD compressor, please configurate building script based on your micro arch(build_scripts/build_mgard_<micro_arch>_hopper.sh)
To build and test the framework:

```
# clone, build all base compressor and then build our multi-component compressor
./build_script.sh 8

# test cpu compressors
./run_script_cpu.sh

# test gpu mgard compressor
./run_script_gpu_mgard.sh

# test gpu zfp compressor
./run_script_gpu_mgard.sh
```

After execution of each runscript, you will find 2 result files for each compressor under Multi-Component-Compression-Framework/

1. result_<compressor_name>.csv
2. result_reconstruct_bound_<compressor_name>.csv

**File 1**
Each result section begins with **Results for file**:, indicating the dataset being tested. The subsequent table contains per-component compression information, with four columns:

- Index: The component ID (starting from 1).
- Tolerance: The fixed-rate (or error bound) used for that component.
- Time(s): The time taken to compress and decompress this component (in seconds).
- CompressedSize(Bytes): The size of the compressed data for this component (in bytes).
At the end of each section, the line Refactor Time shows the total time spent on compressing and decompressing all components of the dataset

**How to read result from file 1**
- The compressed size per component allows you to observe how different tolerance levels affect compression efficiency.
- The total Refactor Time provides an estimate of the end-to-end performance cost.

**File 2**
Each result section begins with **Results for file**:, indicating the dataset being tested. The subsequent table contains per-component compression information, with 3 columns:

- Index: The component ID (starting from 1).
- Tolerance: The fixed-rate (or error bound) used for that component.
- Decompression Time(s): The incremental time taken to progressive decompress to satisfy current tolerance.

**How to read result from file 2**
- The decompression time per component allows you to observe the incremental time to progressive reconstruct to current tolerance.

