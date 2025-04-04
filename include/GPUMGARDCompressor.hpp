#ifndef GPU_MGARD_COMPRESSOR_HPP
#define GPU_MGARD_COMPRESSOR_HPP

#include "GPUCompressor.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mgard/compress_x.hpp"

template <typename T>
class GPUMGARDCompressor : public GPUCompressor<T>
{
public:
  GPUMGARDCompressor() {}

  ~GPUMGARDCompressor() {}

  bool compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                T *original_data, void *&compressed_data,
                size_t &compressed_size)
  {
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;
    // config.dev_type = mgard_x::device_type::SERIAL;

    double s;
    s = std::numeric_limits<float>::infinity();

    mgard_x::data_type dtype;
    if (std::is_same<T, double>::value)
    {
      dtype = mgard_x::data_type::Double;
    }
    else if (std::is_same<T, float>::value)
    {
      dtype = mgard_x::data_type::Float;
    }
    else
    {
      std::cout << "wrong dtype\n";
      exit(-1);
    }

    // By default we use ABS error and L-infinity norm
    mgard_x::compress_status_type status = mgard_x::compress(
        D, dtype, shape, tol, s, mgard_x::error_bound_type::ABS, original_data,
        compressed_data, compressed_size, config, false);

    if (status != mgard_x::compress_status_type::Success)
    {
      std::cerr << "MGARD compress failed with status "
                << static_cast<int>(status) << std::endl;
      return false;
    }
    return true;
  }

  bool decompress(const void *compressed_data, size_t compressed_size,
                  void *&decompressed_data)
  {
    mgard_x::Config config;
    config.dev_type = mgard_x::device_type::CUDA;
    // config.dev_type = mgard_x::device_type::SERIAL;

    mgard_x::compress_status_type status = mgard_x::decompress(
        compressed_data, compressed_size, decompressed_data, config, false);
    if (status != mgard_x::compress_status_type::Success)
    {
      std::cerr << "MGARD decompress failed with status "
                << static_cast<int>(status) << std::endl;
      return false;
    }
    return true;
  }
};

#endif
// GPU_MGARD_COMPRESSOR_HPP
