#ifndef PROGRESSIVE_COMPRESSOR_HPP
#define PROGRESSIVE_COMPRESSOR_HPP

#include "GPUCompressor.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

class ProgressiveCompressor {
public:
  struct Component {
    void *compressed_data;
    size_t compressed_size;
    double tol;
  };

  ProgressiveCompressor(GPUCompressor *compressor)
      : compressor_(compressor), num_components_(0), xTilde_(nullptr), n_(0) {}

  ~ProgressiveCompressor() {
    for (int i = 0; i < num_components_; i++) {
      if (components_[i].compressed_data != nullptr) {
        delete[] static_cast<unsigned char *>(components_[i].compressed_data);
      }
    }

    if (xTilde_) {
      delete[] static_cast<char *>(xTilde_);
    }
  }

  bool compressData(int D, const int *shape, DataType dataType,
                    const void *original_data, const double *toleranceList,
                    int nComponents) {
    std::vector<int> shape_vec;
    for (int i = 0; i < D; i++) {
      shape_vec.push_back(shape[i]);
    }

    n_ = 1;
    for (int i = 0; i < D; i++) {
      n_ *= shape[i];
    }

    if (dataType == DataType::DOUBLE) {
      xTilde_ = new double[n_];
      double *xTildeDouble = static_cast<double *>(xTilde_);
      for (size_t i = 0; i < n_; i++) {
        xTildeDouble[i] = 0.0;
      }
    } else if (dataType == DataType::FLOAT) {
      xTilde_ = new float[n_];
      float *xTildeFloat = static_cast<float *>(xTilde_);
      for (size_t i = 0; i < n_; i++) {
        xTildeFloat[i] = 0.0f;
      }
    } else {
      throw std::runtime_error("Unsupported data type in compressData.");
    }

    components_.resize(nComponents);
    num_components_ = nComponents;

    for (int compIdx = 0; compIdx < nComponents; compIdx++) {
      double tol = toleranceList[compIdx];

      if (dataType == DataType::DOUBLE) {
        double *error = new double[n_];
        const double *orig = static_cast<const double *>(original_data);
        double *xTildeDouble = static_cast<double *>(xTilde_);
        for (size_t j = 0; j < n_; j++) {
          error[j] = orig[j] - xTildeDouble[j];
        }
        void *comp_data = nullptr;
        size_t comp_size = 0;
        bool ok = compressor_->compress(D, shape_vec, tol, dataType, error,
                                        comp_data, comp_size);
        delete[] error;
        if (!ok) {
          std::cerr << "Compression failed at component " << compIdx + 1
                    << std::endl;
          return false;
        }
        components_[compIdx].compressed_data = comp_data;
        components_[compIdx].compressed_size = comp_size;
        components_[compIdx].tol = tol;

        void *decompressed = nullptr;
        ok = compressor_->decompress(comp_data, comp_size, decompressed);
        if (!ok || decompressed == nullptr) {
          std::cerr << "Decompression failed at component " << compIdx + 1
                    << std::endl;
          return false;
        }
        double *xTildeDoubleAfter = static_cast<double *>(xTilde_);
        const double *origDouble = static_cast<const double *>(original_data);
        double *decompressed_data = static_cast<double *>(decompressed);
        double maxErr = 0.0;
        for (size_t j = 0; j < n_; j++) {
          xTildeDoubleAfter[j] += decompressed_data[j];
          double diff = std::fabs(origDouble[j] - xTildeDoubleAfter[j]);
          if (diff > maxErr)
            maxErr = diff;
        }
        delete[] decompressed_data;
        double ratio = maxErr / tol;
        std::cout << "Component " << compIdx + 1 << ": max error = " << maxErr
                  << ", tol = " << tol << ", ratio = " << ratio << std::endl;
      } else if (dataType == DataType::FLOAT) {
        float *error = new float[n_];
        const float *orig = static_cast<const float *>(original_data);
        float *xTildeFloat = static_cast<float *>(xTilde_);
        for (size_t j = 0; j < n_; j++) {
          error[j] = orig[j] - xTildeFloat[j];
        }
        void *comp_data = nullptr;
        size_t comp_size = 0;
        bool ok = compressor_->compress(D, shape_vec, tol, dataType, error,
                                        comp_data, comp_size);
        delete[] error;
        if (!ok) {
          std::cerr << "Compression failed at component " << compIdx + 1
                    << std::endl;
          return false;
        }
        components_[compIdx].compressed_data = comp_data;
        components_[compIdx].compressed_size = comp_size;
        components_[compIdx].tol = tol;

        void *decompressed = nullptr;
        ok = compressor_->decompress(comp_data, comp_size, decompressed);
        if (!ok || decompressed == nullptr) {
          std::cerr << "Decompression failed at component " << compIdx + 1
                    << std::endl;
          return false;
        }
        float *xTildeFloatAfter = static_cast<float *>(xTilde_);
        const float *origFloat = static_cast<const float *>(original_data);

        double *decompressed_data = static_cast<double *>(decompressed);
        double maxErr = 0.0;
        for (size_t j = 0; j < n_; j++) {
          xTildeFloatAfter[j] += static_cast<float>(decompressed_data[j]);
          float diff = std::fabs(origFloat[j] - xTildeFloatAfter[j]);
          if (diff > maxErr)
            maxErr = diff;
        }
        delete[] decompressed_data;
        double ratio = maxErr / tol;
        std::cout << "Component " << compIdx + 1 << ": max error = " << maxErr
                  << ", tol = " << tol << ", ratio = " << ratio << std::endl;
      } else {
        std::cerr << "Unsupported data type." << std::endl;
        return false;
      }
    }
    return true;
  }

  double *reconstructData(int m) {
    if (m > num_components_)
      m = num_components_;
    double *result = new double[n_];
    for (size_t j = 0; j < n_; j++) {
      result[j] = 0.0;
    }
    for (int i = 0; i < m; i++) {
      std::cout << "Componet " << i + 1 << " has been used!" << std::endl;
      void *decompressed = nullptr;
      bool ok =
          compressor_->decompress(components_[i].compressed_data,
                                  components_[i].compressed_size, decompressed);
      if (!ok || decompressed == nullptr) {
        std::cerr << "Decompression failed in reconstructData at component "
                  << i + 1 << std::endl;
        delete[] result;
        return nullptr;
      }
      double *decompressed_data = static_cast<double *>(decompressed);
      for (size_t j = 0; j < n_; j++) {
        result[j] += decompressed_data[j];
      }
      delete[] decompressed_data;
    }
    return result;
  }

  int getNumComponents() const { return num_components_; }

private:
  GPUCompressor *compressor_;
  std::vector<Component> components_;
  int num_components_;
  void *xTilde_;
  size_t n_;
};

#endif // PROGRESSIVE_COMPRESSOR_HPP
