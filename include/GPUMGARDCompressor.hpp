#ifndef GPU_MGARD_COMPRESSOR_HPP
#define GPU_MGARD_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mgard/compress_x.hpp"
#include <cuda_runtime.h>

template <typename T> class GPUMGARDCompressor : public GeneralCompressor<T> {
public:
  double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size, bool include_copy_time) override {
    auto t0 = std::chrono::high_resolution_clock::now();
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;

    double s = std::numeric_limits<float>::infinity();

    mgard_x::data_type dtype;
    if (std::is_same<T, double>::value) {
      dtype = mgard_x::data_type::Double;
    } else if (std::is_same<T, float>::value) {
      dtype = mgard_x::data_type::Float;
    } else {
      std::cerr << "Unsupported data type for MGARD\n";
      std::exit(EXIT_FAILURE);
    }

    size_t original_size = 1;
    for (mgard_x::DIM i = 0; i < D; i++) {
      original_size *= shape[i];
    }

    T *d_original_data;
    cudaMalloc((void **)&d_original_data, original_size * sizeof(T));
    cudaMemcpy(d_original_data, original_data, original_size * sizeof(T),
               cudaMemcpyHostToDevice);
    void *compressed_array_gpu = nullptr;
    // If needed, we need expand this buffer size
    cudaMalloc((void **)&compressed_array_gpu, original_size * sizeof(T) + 1e9);
    compressed_size = original_size + 1e9;

    auto t1 = std::chrono::high_resolution_clock::now();

    mgard_x::compress_status_type status = mgard_x::compress(
        D, dtype, shape, tol, s, mgard_x::error_bound_type::ABS,
        d_original_data, compressed_array_gpu, compressed_size, config, true);

    auto t2 = std::chrono::high_resolution_clock::now();

    if (status != mgard_x::compress_status_type::Success) {
      std::cerr << "MGARD compress failed with status "
                << static_cast<int>(status) << std::endl;
      std::exit(EXIT_FAILURE);
    }

    compressed_data = malloc(compressed_size);
    cudaMemcpy(compressed_data, compressed_array_gpu, compressed_size,
               cudaMemcpyDeviceToHost);
    cudaFree(d_original_data);
    cudaFree(compressed_array_gpu);

    double elapsed_time;
    if (include_copy_time) {
      elapsed_time = std::chrono::duration<double>(t2 - t0).count();
    } else {
      elapsed_time = std::chrono::duration<double>(t2 - t1).count();
    }

    return elapsed_time;
  }

  double decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                    double tol, void *compressed_data, size_t compressed_size,
                    void *&decompressed_data, bool include_copy_time) override {
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;

    size_t original_size = 1;
    for (mgard_x::DIM i = 0; i < D; i++) {
      original_size *= shape[i];
    }

    void *compressed_array_gpu;
    cudaMalloc((void **)&compressed_array_gpu, compressed_size);
    cudaMemcpy(compressed_array_gpu, compressed_data, compressed_size,
               cudaMemcpyHostToDevice);
    void *d_decompressed_data;
    cudaMalloc((void **)&d_decompressed_data, original_size * sizeof(T));

    auto t0 = std::chrono::high_resolution_clock::now();

    mgard_x::compress_status_type status =
        mgard_x::decompress(compressed_array_gpu, compressed_size,
                            d_decompressed_data, config, true);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (status != mgard_x::compress_status_type::Success) {
      std::cerr << "MGARD decompress failed with status "
                << static_cast<int>(status) << std::endl;
      std::exit(EXIT_FAILURE);
    }

    decompressed_data = malloc(original_size * sizeof(T));
    cudaMemcpy(decompressed_data, d_decompressed_data,
               original_size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(compressed_array_gpu);
    cudaFree(d_decompressed_data);

    auto t2 = std::chrono::high_resolution_clock::now();

    double elapsed_time;
    if (include_copy_time) {
      elapsed_time = std::chrono::duration<double>(t2 - t0).count();
    } else {
      elapsed_time = std::chrono::duration<double>(t1 - t0).count();
    }

    return elapsed_time;
  }
};

#endif // GPU_MGARD_COMPRESSOR_HPP
