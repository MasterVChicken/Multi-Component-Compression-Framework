#ifndef GPU_MGARD_COMPRESSOR_HPP
#define GPU_MGARD_COMPRESSOR_HPP

#include "GPUCompressor.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mgard/compress_x.hpp"

class GPUMGARDCompressor : public GPUCompressor {
public:
  GPUMGARDCompressor() {}

  virtual ~GPUMGARDCompressor() {}

  bool compress(int D, std::vector<int> shape, double tol, DataType dataType,
                const void *original_data, void *&compressed_data,
                size_t &compressed_size) {
    mgard_x::data_type dtype;
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;

    std::vector<mgard_x::SIZE> mgard_shape;
    for (int i = 0; i < shape.size(); i++) {
      mgard_shape.push_back(static_cast<mgard_x::SIZE>(shape[i]));
    }

    double s;
    if (dataType == DataType::FLOAT) {
      dtype = mgard_x::data_type::Float;
      s = std::numeric_limits<float>::infinity();
    } else if (dataType == DataType::DOUBLE) {
      dtype = mgard_x::data_type::Double;
      s = std::numeric_limits<double>::infinity();
    }

    // By default we use ABS error and L-infinity norm
    mgard_x::compress_status_type status = mgard_x::compress(
        D, dtype, mgard_shape, tol, s, mgard_x::error_bound_type::ABS,
        original_data, compressed_data, compressed_size, config, false);

    if (status != mgard_x::compress_status_type::Success) {
      std::cerr << "MGARD compress failed with status "
                << static_cast<int>(status) << std::endl;
      return false;
    }
    return true;
  }

  bool decompress(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data) {
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;

    mgard_x::compress_status_type status = mgard_x::decompress(
        compressed_data, compressed_size, decompressed_data, config, false);
    if (status != mgard_x::compress_status_type::Success) {
      std::cerr << "MGARD decompress failed with status "
                << static_cast<int>(status) << std::endl;
      return false;
    }
    return true;
  }
};

#endif
// GPU_MGARD_COMPRESSOR_HPP
