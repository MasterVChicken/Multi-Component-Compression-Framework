#ifndef GPU_SZ_COMPRESSOR_HPP
#define GPU_SZ_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include <cuSZp.h>
#include <cuda_runtime.h>

template <typename T> class GPUSZCompressor : public GeneralCompressor<T> {
public:
  double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size, bool include_copy_time) override {
    auto t0 = std::chrono::high_resolution_clock::now();
    cuszp_type_t dataType;
    // or we can use CUSZP_MODE_OUTLIER
    cuszp_mode_t encodingMode = CUSZP_MODE_PLAIN;

    if (std::is_same<T, double>::value) {
      dataType = CUSZP_TYPE_DOUBLE;
    } else if (std::is_same<T, float>::value) {
      dataType = CUSZP_TYPE_FLOAT;
    } else {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    size_t data_size = 1;
    for (int i = 0; i < D; i++) {
      data_size *= shape[i];
    }

    T *d_oriData;
    unsigned char *d_cmpBytes;
    cudaMalloc((void **)&d_oriData, data_size * sizeof(T));
    cudaMemcpy(d_oriData, original_data, sizeof(T) * data_size,
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_cmpBytes, data_size * sizeof(T));

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t1 = std::chrono::high_resolution_clock::now();

    // d_oriData and d_cmpBytes are device pointers
    cuSZp_compress(d_oriData, d_cmpBytes, data_size, &compressed_size, tol,
                   dataType, encodingMode, stream);

    auto t2 = std::chrono::high_resolution_clock::now();

    compressed_data = malloc(compressed_size);
    cudaMemcpy(compressed_data, d_cmpBytes, compressed_size,
               cudaMemcpyDeviceToHost);
    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

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

    cuszp_type_t dataType;
    // or we can use CUSZP_MODE_OUTLIER
    cuszp_mode_t encodingMode = CUSZP_MODE_PLAIN;

    if (std::is_same<T, double>::value) {
      dataType = CUSZP_TYPE_DOUBLE;
    } else if (std::is_same<T, float>::value) {
      dataType = CUSZP_TYPE_FLOAT;
    } else {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    size_t data_size = 1;
    for (int i = 0; i < D; i++) {
      data_size *= shape[i];
    }

    T *d_decData;
    unsigned char *d_cmpBytes;
    cudaMalloc((void **)&d_decData, data_size * sizeof(T));
    cudaMalloc((void **)&d_cmpBytes, compressed_size);

    cudaMemcpy(d_cmpBytes, compressed_data, compressed_size,
               cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto t0 = std::chrono::high_resolution_clock::now();

    // d_cmpBytes and d_decData are device pointers
    cuSZp_decompress(d_decData, d_cmpBytes, data_size, compressed_size, tol,
                     dataType, encodingMode, stream);

    auto t1 = std::chrono::high_resolution_clock::now();

    decompressed_data = malloc(data_size * sizeof(T));
    cudaMemcpy(decompressed_data, d_decData, data_size * sizeof(T),
               cudaMemcpyDeviceToHost);

    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

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
// GPU_SZ_COMPRESSOR_HPP
