#ifndef GPU_ZFP_COMPRESSOR_HPP
#define GPU_ZFP_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "zfp.h"
#include <chrono>
#include <cmath>
#include <cstring> // for memcpy
#include <cuda_runtime.h>
#include <iostream>

template <typename T> class GPUZFPCompressor : public GeneralCompressor<T> {
public:
  // Store fixed-rate data on the head of file

  double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size, bool include_copy_time) override {

    auto t0 = std::chrono::high_resolution_clock::now();
    size_t original_size = 1;
    for (mgard_x::DIM i = 0; i < D; i++)
      original_size *= shape[i];

    zfp_type type;
    if (std::is_same<T, double>::value)
      type = zfp_type_double;
    else if (std::is_same<T, float>::value)
      type = zfp_type_float;
    else {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    // get current max value
    T max_abs = 0;
    for (size_t i = 0; i < original_size; i++) {
      T abs_val = std::fabs(original_data[i]);
      if (abs_val > max_abs)
        max_abs = abs_val;
    }

    // 3.14 works for temperature.f32 in NYX dataset
    // double offset = 3.14;

    // 2.32 works for Pf48.bin.f32 in Hurricane ISABEL dataset
    double offset = 2.32;

    // 2.41 works for PRES-98x1200x1200.f32 in SCALE-LETKF
    // double offset = 2.41;

    // works for velocityz.d64 in Miranda
    // double offset = 2.73;
    double computed_rate = std::log2(max_abs / tol) + offset;
    double fixed_rate = computed_rate < 1.0 ? 1.0 : computed_rate;
    // debug info
    // std::cout << "max_abs = " << max_abs << ", tol = " << tol
    //           << ", fixed_rate = " << fixed_rate << std::endl;

    T *d_original_data = nullptr;
    cudaMalloc(&d_original_data, sizeof(T) * original_size);
    cudaMemcpy(d_original_data, original_data, sizeof(T) * original_size,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    zfp_field *field = nullptr;
    if (D == 1)
      field = zfp_field_1d(d_original_data, type, shape[0]);
    else if (D == 2)
      field = zfp_field_2d(d_original_data, type, shape[1], shape[0]);
    else if (D == 3)
      field = zfp_field_3d(d_original_data, type, shape[2], shape[1], shape[0]);
    else if (D == 4)
      field = zfp_field_4d(d_original_data, type, shape[3], shape[2], shape[1],
                           shape[0]);
    else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, fixed_rate, type, zfp_field_dimensionality(field),
                        zfp_false);

    // size_t bufsize = zfp_stream_maximum_size(zfp, field);
    size_t bufsize = original_size * sizeof(T) * 2;
    void *d_compressed_data = nullptr;
    cudaError_t err = cudaMalloc(&d_compressed_data, bufsize);
    // std::cout << "bufsize: " << bufsize << std::endl;
    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err)
                << std::endl;
      exit(-1);
    }

    bitstream *stream = stream_open(d_compressed_data, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
      std::cout << "zfp-cuda not available\n";
      exit(-1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    compressed_size = zfp_compress(zfp, field);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    if (compressed_size == 0) {
      std::cout << "zfp-cuda compress error\n";
      exit(-1);
    }

    void *host_buffer = malloc(compressed_size);
    if (!host_buffer) {
      std::cerr << "Failed to allocate host memory for compression."
                << std::endl;
      exit(-1);
    }
    cudaMemcpy(host_buffer, d_compressed_data, compressed_size,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_original_data);
    cudaFree(d_compressed_data);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    // Store fixed_rate
    size_t header_size = sizeof(double);
    void *final_buffer = malloc(header_size + compressed_size);
    if (!final_buffer) {
      std::cerr << "Failed to allocate host memory for header." << std::endl;
      exit(-1);
    }

    memcpy(final_buffer, &fixed_rate, header_size);

    memcpy((char *)final_buffer + header_size, host_buffer, compressed_size);
    free(host_buffer);

    compressed_size += header_size;
    compressed_data = final_buffer;

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
    size_t original_size = 1;
    for (mgard_x::DIM i = 0; i < D; i++)
      original_size *= shape[i];

    zfp_type type;
    if (std::is_same<T, double>::value)
      type = zfp_type_double;
    else if (std::is_same<T, float>::value)
      type = zfp_type_float;
    else {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    // Get saved fixed_rate from header
    size_t header_size = sizeof(double);
    if (compressed_size < header_size) {
      std::cerr << "Compressed data too small for header." << std::endl;
      exit(-1);
    }
    double stored_fixed_rate;
    memcpy(&stored_fixed_rate, compressed_data, header_size);
    void *real_compressed_data = (char *)compressed_data + header_size;
    size_t real_compressed_size = compressed_size - header_size;

    // debug info
    // std::cout << "Decompress: using stored fixed_rate = " <<
    // stored_fixed_rate
    //           << std::endl;

    T *d_decompressed_data = nullptr;
    cudaMalloc(&d_decompressed_data, sizeof(T) * original_size);
    uint8_t *d_compressed_data = nullptr;
    cudaMalloc(&d_compressed_data, real_compressed_size);
    cudaMemcpy(d_compressed_data, real_compressed_data, real_compressed_size,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    zfp_field *field = nullptr;
    if (D == 1)
      field = zfp_field_1d(d_decompressed_data, type, shape[0]);
    else if (D == 2)
      field = zfp_field_2d(d_decompressed_data, type, shape[1], shape[0]);
    else if (D == 3)
      field =
          zfp_field_3d(d_decompressed_data, type, shape[2], shape[1], shape[0]);
    else if (D == 4)
      field = zfp_field_4d(d_decompressed_data, type, shape[3], shape[2],
                           shape[1], shape[0]);
    else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(NULL);

    zfp_stream_set_rate(zfp, stored_fixed_rate, type,
                        zfp_field_dimensionality(field), zfp_false);

    bitstream *stream = stream_open(d_compressed_data, real_compressed_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (!zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
      std::cout << "zfp-cuda not available\n";
      exit(-1);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int status = zfp_decompress(zfp, field);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!status) {
      std::cout << "zfp-cuda decompress error\n";
      exit(-1);
    }

    decompressed_data = new T[original_size];
    cudaMemcpy(decompressed_data, d_decompressed_data,
               sizeof(T) * original_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_decompressed_data);
    cudaFree(d_compressed_data);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

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

#endif
