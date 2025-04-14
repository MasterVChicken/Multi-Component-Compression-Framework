#ifndef CPU_ZFP_COMPRESSOR_HPP
#define CPU_ZFP_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "zfp.h"
#include <cstdlib>
#include <iostream>
#include <type_traits>

// OPENMP

template <typename T> class CPUZFPCompressor : public GeneralCompressor<T> {
public:
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

    zfp_field *field;
    if (D == 1) {
      field = zfp_field_1d(original_data, type, shape[0]);
    } else if (D == 2) {
      field = zfp_field_2d(original_data, type, shape[1], shape[0]);
    } else if (D == 3) {
      field = zfp_field_3d(original_data, type, shape[2], shape[1], shape[0]);
    } else if (D == 4) {
      field = zfp_field_4d(original_data, type, shape[3], shape[2], shape[1],
                           shape[0]);
    } else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tol);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void *buffer = malloc(bufsize);
    if (!buffer) {
      std::cerr << "Failed to allocate memory for compression." << std::endl;
      exit(-1);
    }
    compressed_data = buffer;
    compressed_size = bufsize;

    bitstream *stream = stream_open(compressed_data, compressed_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (!zfp_stream_set_execution(zfp, zfp_exec_omp)) {
      std::cout << "zfp-openmp not available\n";
      exit(-1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    compressed_size = zfp_compress(zfp, field);

    auto t2 = std::chrono::high_resolution_clock::now();

    if (compressed_size == 0) {
      std::cout << "zfp compress error\n";
      exit(-1);
    }

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

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

    decompressed_data = new T[original_size];

    zfp_type type;
    if (std::is_same<T, double>::value)
      type = zfp_type_double;
    else if (std::is_same<T, float>::value)
      type = zfp_type_float;
    else {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    zfp_field *field;
    if (D == 1)
      field = zfp_field_1d(static_cast<T *>(decompressed_data), type, shape[0]);
    else if (D == 2)
      field = zfp_field_2d(static_cast<T *>(decompressed_data), type, shape[1],
                           shape[0]);
    else if (D == 3)
      field = zfp_field_3d(static_cast<T *>(decompressed_data), type, shape[2],
                           shape[1], shape[0]);
    else if (D == 4)
      field = zfp_field_4d(static_cast<T *>(decompressed_data), type, shape[3],
                           shape[2], shape[1], shape[0]);
    else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    zfp_stream *zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tol);

    bitstream *stream = stream_open(compressed_data, compressed_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (!zfp_stream_set_execution(zfp, zfp_exec_serial)) {
      std::cout << "zfp-serial not available\n";
      exit(-1);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int status = zfp_decompress(zfp, field);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (!status) {
      std::cout << "zfp decompress error\n";
      exit(-1);
    }

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

    return elapsed_time;
  }
};

#endif
