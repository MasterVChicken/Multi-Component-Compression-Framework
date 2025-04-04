#ifndef GPU_COMPRESSOR_HPP
#define GPU_COMPRESSOR_HPP

#include "mgard/compress_x.hpp"

// By default we only apply L-infinity norm and ABS error mode
// So we omit these params

// Abstract GPU compressor interface
template <typename T>
class GPUCompressor
{
public:
  virtual bool compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                        double tol, T *original_data, void *&compressed_data,
                        size_t &compressed_size) = 0;

  virtual bool decompress(const void *compressed_data, size_t compressed_size,
                          void *&decompressed_data) = 0;
};

#endif
// GPU_COMPRESSOR_HPP
