# Multi-Component-Compression-Framework

GPU Compressor: MGARD(1.5.2), ZFP(1.0.0)

CPU Compressor: ZFP(1.0.0), SZ3(v3.1.7)

Make sure you adjust configuration and data path first.

To build and test the framework:

```
# build
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

**How to read result from file 1**
- The decompression time per component allows you to observe the incremental time to progressive reconstruct to current tolerance.

