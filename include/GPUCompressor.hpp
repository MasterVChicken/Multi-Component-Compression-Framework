#ifndef GPU_COMPRESSOR_H
#define GPU_COMPRESSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

enum class DataType { FLOAT, DOUBLE };

// Abstract GPU compressor interface
class GPUCompressor {
public:
  virtual ~GPUCompressor() {}
  virtual bool compress(int D, std::vector<int> shape, double tol,
                        DataType dataType, const void *original_data,
                        void *&compressed_data, size_t &compressed_size) = 0;

  virtual bool decompress(const void *compressed_data, size_t compressed_size,
                          void *&decompressed_data) = 0;
};

#endif
// GPU_COMPRESSOR_H
