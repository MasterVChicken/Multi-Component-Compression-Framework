#ifndef PROGRESSIVE_COMPRESSOR_CPU_HPP
#define PROGRESSIVE_COMPRESSOR_CPU_HPP

#include "GeneralCompressor.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void launchComputeErrorCPU(const T *h_orig, T *h_xTilde, T *h_error, size_t n) {
#pragma omp parallel for
  for (size_t idx = 0; idx < n; idx++) {
    h_error[idx] = h_orig[idx] - h_xTilde[idx];
  }
}

template <typename T>
void launchAccumulateCPU(T *h_result, T *h_decompressed, size_t n) {
#pragma omp parallel for
  for (size_t idx = 0; idx < n; idx++) {
    h_result[idx] += h_decompressed[idx];
  }
}

template <typename T> class ProgressiveCompressor {
public:
  struct Component {
    void *compressed_data;
    size_t compressed_size;
    double tol;
  };

  ProgressiveCompressor(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                        GeneralCompressor<T> *compressor)
      : D_(D), shape_(shape), compressor_(compressor), num_components_(0),
        xTilde_(nullptr), n_(0) {}

  ~ProgressiveCompressor() {
    for (int i = 0; i < num_components_; i++) {
      if (components_[i].compressed_data != nullptr) {
        free(components_[i].compressed_data);
      }
    }
    if (xTilde_) {
      delete[] xTilde_;
    }
  }

  bool compressData(const T *original_data, const double *relToleranceList,
                    int nComponents) {
    n_ = 1;
    for (int i = 0; i < D_; i++) {
      n_ *= shape_[i];
    }

    xTilde_ = new T[n_];
    for (size_t i = 0; i < n_; i++) {
      xTilde_[i] = 0.0;
    }

    components_.resize(nComponents);
    compressTimes_.resize(nComponents, 0.0);
    num_components_ = nComponents;

    T l_inf_norm = 0.0;
    for (size_t i = 0; i < n_; i++) {
      l_inf_norm = std::max(l_inf_norm, std::fabs(original_data[i]));
    }

    for (int compIdx = 0; compIdx < nComponents; compIdx++) {
      double rel_tol = relToleranceList[compIdx];
      T *error = new T[n_];

      launchComputeErrorCPU<T>(original_data, xTilde_, error, n_);

      double abs_tol = rel_tol * l_inf_norm;

      void *comp_data = nullptr;
      size_t comp_size = 0;
      double comp_time = compressor_->compress(
          D_, shape_, abs_tol, error, comp_data, comp_size, compIdx == 0);

      delete[] error;

      components_[compIdx].compressed_data = comp_data;
      components_[compIdx].compressed_size = comp_size;
      components_[compIdx].tol = rel_tol;

      void *decompressed = nullptr;
      double decomp_time =
          compressor_->decompress(D_, shape_, abs_tol, comp_data, comp_size,
                                  decompressed, compIdx == nComponents - 1);

      compressTimes_[compIdx] = comp_time + decomp_time;

      T *decompressed_data = static_cast<T *>(decompressed);
      for (size_t j = 0; j < n_; j++) {
        xTilde_[j] += decompressed_data[j];
      }

      delete[] decompressed_data;
    }
    return true;
  }

  T *reconstructData(int m) {
    if (m > num_components_)
      m = num_components_;
    T *result = new T[n_];
    std::fill(result, result + n_, 0.0);

    component_decomp_times_.clear();
    component_decomp_times_.resize(m, 0.0);

    for (int i = 0; i < m; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      void *decompressed = nullptr;
      double tol = components_[i].tol;
      compressor_->decompress(D_, shape_, tol, components_[i].compressed_data,
                              components_[i].compressed_size, decompressed,
                              false);

      if (!decompressed) {
        std::cerr << "[Warning] decompress() returned nullptr at component "
                  << (i + 1) << std::endl;
        continue;
      }
      T *decompressed_data = static_cast<T *>(decompressed);
      launchAccumulateCPU<T>(result, decompressed_data, n_);
      delete[] decompressed_data;

      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      component_decomp_times_[i] = dt;
      
      if (m == num_components_) {
        std::cout << "Component " << i + 1
                  << ": incremental decompression time = " << dt << " s"
                  << std::endl;
      }
    }
    return result;
  }

  const std::vector<double> &getComponentDecompressionTimes() const {
    return component_decomp_times_;
  }

  int getNumComponents() const { return num_components_; }

  size_t getCompressedSize(int componentIdx) const {
    if (componentIdx < 0 || componentIdx >= num_components_)
      return 0;
    return components_[componentIdx].compressed_size;
  }

  double getComponentCompressTime(int idx) const {
    if (idx < 0 || idx >= compressTimes_.size())
      return 0;
    return compressTimes_[idx];
  }

private:
  GeneralCompressor<T> *compressor_;
  std::vector<Component> components_;
  std::vector<double> compressTimes_;
  int num_components_;
  T *xTilde_;
  size_t n_;
  mgard_x::DIM D_;
  std::vector<mgard_x::SIZE> shape_;

  std::vector<double> component_decomp_times_;
};

#endif // PROGRESSIVE_COMPRESSOR_CPU_HPP
