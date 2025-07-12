#ifndef GPU_ZFP_COMPRESSOR_HPP
#define GPU_ZFP_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "zfp.h"
#include <chrono>
#include <cmath>
#include <cstring> // for memcpy
#include <iostream>

template <typename T>
class GPUZFPCompressor : public GeneralCompressor<T> {
public:
  // Host-only + CUDA backend fixed-rate compression
  double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                  double fixed_rate, T *original_data, void *&compressed_data,
                  size_t &compressed_size, bool include_copy_time) override {
// std::cout << "Fixed Rate: " << fixed_rate << std::endl;
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
      std::cerr << "Unsupported data type\n";
      exit(-1);
    }

    zfp_field *field = nullptr;
    if (D == 1)
      field = zfp_field_1d(original_data, type, shape[0]);
    else if (D == 2)
      field = zfp_field_2d(original_data, type, shape[1], shape[0]);
    else if (D == 3)
      field = zfp_field_3d(original_data, type, shape[2], shape[1], shape[0]);
    else if (D == 4)
      field = zfp_field_4d(original_data, type, shape[3], shape[2], shape[1], shape[0]);
    else {
      std::cerr << "Unsupported dimension\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zfp, fixed_rate, type, zfp_field_dimensionality(field), zfp_false);
    // std::cout << "Fixed rate: " << fixed_rate << std::endl;
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    // std::cout << bufsize << std::endl;
    void *host_buffer = malloc(bufsize);
    if (!host_buffer) {
      std::cerr << "Failed to allocate host buffer\n";
      exit(-1);
    }

    bitstream *stream = stream_open(host_buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // 强制启用 CUDA backend（可选）
    zfp_stream_set_execution(zfp, zfp_exec_cuda);

    auto t1 = std::chrono::high_resolution_clock::now();
    compressed_size = zfp_compress(zfp, field);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (compressed_size == 0) {
      std::cerr << "ZFP compression failed\n";
      exit(-1);
    }

    compressed_data = malloc(compressed_size);
    if (!compressed_data) {
      std::cerr << "Failed to allocate output buffer\n";
      exit(-1);
    }
    std::memcpy(compressed_data, host_buffer, compressed_size);

    // Clean up
    free(host_buffer);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return include_copy_time
               ? std::chrono::duration<double>(t2 - t0).count()
               : std::chrono::duration<double>(t2 - t1).count();
  }

  double decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                    double fixed_rate, void *compressed_data,
                    size_t compressed_size, void *&decompressed_data,
                    bool include_copy_time) override {

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
      std::cerr << "Unsupported data type\n";
      exit(-1);
    }

    decompressed_data = malloc(original_size * sizeof(T));
    if (!decompressed_data) {
      std::cerr << "Failed to allocate decompressed buffer\n";
      exit(-1);
    }

    zfp_field *field = nullptr;
    if (D == 1)
      field = zfp_field_1d(decompressed_data, type, shape[0]);
    else if (D == 2)
      field = zfp_field_2d(decompressed_data, type, shape[1], shape[0]);
    else if (D == 3)
      field = zfp_field_3d(decompressed_data, type, shape[2], shape[1], shape[0]);
    else if (D == 4)
      field = zfp_field_4d(decompressed_data, type, shape[3], shape[2], shape[1], shape[0]);
    else {
      std::cerr << "Unsupported dimension\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zfp, fixed_rate, type, zfp_field_dimensionality(field), zfp_false);

    bitstream *stream = stream_open(compressed_data, compressed_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    zfp_stream_set_execution(zfp, zfp_exec_cuda);  // optional but recommended

    auto t1 = std::chrono::high_resolution_clock::now();
    int status = zfp_decompress(zfp, field);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (!status) {
      std::cerr << "ZFP decompression failed\n";
      exit(-1);
    }

    // Clean up
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return include_copy_time
               ? std::chrono::duration<double>(t2 - t0).count()
               : std::chrono::duration<double>(t2 - t1).count();
  }
};

#endif
