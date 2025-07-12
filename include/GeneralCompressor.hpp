#ifndef GPU_COMPRESSOR_HPP
#define GPU_COMPRESSOR_HPP

#include "mgard/compress_x.hpp"
#include <chrono>

// By default we only apply L-infinity norm and ABS error mode
// So we omit these params

// Abstract GPU compressor interface
template <typename T> class GeneralCompressor {
public:
  virtual double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                          double tol, T *original_data, void *&compressed_data,
                          size_t &compressed_size, bool include_copy_time) = 0;

  virtual double decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                            double tol, void *compressed_data,
                            size_t compressed_size, void *&decompressed_data,
                            bool include_copy_time) = 0;
};

#endif
// GPU_COMPRESSOR_HPP