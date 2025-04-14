// #ifndef PROGRESSIVE_COMPRESSOR_HPP
// #define PROGRESSIVE_COMPRESSOR_HPP

// #include "GeneralCompressor.hpp"
// #include <cmath>
// #include <cstddef>
// #include <cstdlib>
// #include <fstream>
// #include <iostream>
// #include <sstream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// #ifdef ENABLE_CUDA_COMPRESSOR
// #include <cuda_runtime.h>
// #else
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// #endif

// #if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR &&               \
//     defined(__CUDACC__)

// template <typename T>
// __global__ void computeErrorKernel(const T *d_orig, const T *d_xTilde,
//                                    T *d_error, size_t n) {
//   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     d_error[idx] = d_orig[idx] - d_xTilde[idx];
//   }
// }

// template <typename T>
// void launchComputeErrorKernel(const T *h_orig, T *h_xTilde, T *h_error,
//                               size_t n) {
//   T *d_orig = nullptr, *d_xTilde = nullptr, *d_error = nullptr;
//   cudaMalloc(&d_orig, n * sizeof(T));
//   cudaMalloc(&d_xTilde, n * sizeof(T));
//   cudaMalloc(&d_error, n * sizeof(T));

//   cudaMemcpy(d_orig, h_orig, n * sizeof(T), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_xTilde, h_xTilde, n * sizeof(T), cudaMemcpyHostToDevice);

//   int threads = 256;
//   int blocks = (n + threads - 1) / threads;
//   computeErrorKernel<T><<<blocks, threads>>>(d_orig, d_xTilde, d_error, n);
//   cudaDeviceSynchronize();

//   cudaMemcpy(h_error, d_error, n * sizeof(T), cudaMemcpyDeviceToHost);

//   cudaFree(d_orig);
//   cudaFree(d_xTilde);
//   cudaFree(d_error);
// }

// template <typename T>
// __global__ void accumulateKernel(const T *d_decompressed, T *d_result,
//                                  size_t n) {
//   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     d_result[idx] += d_decompressed[idx];
//   }
// }

// template <typename T>
// void launchAccumulateKernel(T *h_result, T *h_decompressed, size_t n) {
//   T *d_result = nullptr, *d_decompressed = nullptr;
//   cudaMalloc(&d_result, n * sizeof(T));
//   cudaMalloc(&d_decompressed, n * sizeof(T));

//   cudaMemcpy(d_result, h_result, n * sizeof(T), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_decompressed, h_decompressed, n * sizeof(T),
//              cudaMemcpyHostToDevice);

//   int threads = 256;
//   int blocks = (n + threads - 1) / threads;
//   accumulateKernel<T><<<blocks, threads>>>(d_decompressed, d_result, n);
//   cudaDeviceSynchronize();

//   cudaMemcpy(h_result, d_result, n * sizeof(T), cudaMemcpyDeviceToHost);

//   cudaFree(d_result);
//   cudaFree(d_decompressed);
// }

// #else
// template <typename T>
// void launchComputeErrorCPU(const T *h_orig, T *h_xTilde, T *h_error, size_t
// n) { #pragma omp parallel for
//   for (size_t idx = 0; idx < n; idx++) {
//     h_error[idx] = h_orig[idx] - h_xTilde[idx];
//   }
// }

// template <typename T>
// void launchAccumulateCPU(T *h_result, T *h_decompressed, size_t n) {
// #pragma omp parallel for
//   for (size_t idx = 0; idx < n; idx++) {
//     h_result[idx] += h_decompressed[idx];
//   }
// }
// #endif

// template <typename T> class ProgressiveCompressor {
// public:
//   struct Component {
//     void *compressed_data;
//     size_t compressed_size;
//     double tol;
//   };

//   ProgressiveCompressor(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
//                         GeneralCompressor<T> *compressor)
//       : D_(D), shape_(shape), compressor_(compressor), num_components_(0),
//         xTilde_(nullptr), n_(0) {}

//   ~ProgressiveCompressor() {
//     for (int i = 0; i < num_components_; i++) {
//       if (components_[i].compressed_data != nullptr) {
//         free(components_[i].compressed_data);
//       }
//     }
//     if (xTilde_) {
//       delete[] xTilde_;
//     }
//   }

//   bool compressData(const T *original_data, const double *toleranceList,
//                     int nComponents) {
//     n_ = 1;
//     for (int i = 0; i < D_; i++) {
//       n_ *= shape_[i];
//     }

//     xTilde_ = new T[n_];
//     for (size_t i = 0; i < n_; i++) {
//       xTilde_[i] = 0.0;
//     }

//     components_.resize(nComponents);
//     compressTimes_.resize(nComponents, 0.0);
//     num_components_ = nComponents;

//     // const double early_stop_threshold = 1e-12;

//     for (int compIdx = 0; compIdx < nComponents; compIdx++) {
//       double tol = toleranceList[compIdx];
//       T *error = new T[n_];

// #if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR &&               \
//     defined(__CUDACC__)
//       launchComputeErrorKernel<T>(original_data, xTilde_, error, n_);
// #else
//       launchComputeErrorCPU<T>(original_data, xTilde_, error, n_);
// #endif

//       T max_residual = 0.0;
//       for (size_t i = 0; i < n_; i++) {
//         max_residual = std::max(max_residual, std::fabs(error[i]));
//       }
//       std::cout << "[Comp " << compIdx + 1
//                 << "] Residual max = " << max_residual << ", tol = " << tol
//                 << std::endl;

//       void *comp_data = nullptr;
//       size_t comp_size = 0;
//       double comp_time = compressor_->compress(
//           D_, shape_, tol, error, comp_data, comp_size, compIdx == 0);

//       delete[] error;

//       components_[compIdx].compressed_data = comp_data;
//       components_[compIdx].compressed_size = comp_size;
//       components_[compIdx].tol = tol;

//       void *decompressed = nullptr;
//       if (compIdx == 0) {
//       }
//       double decomp_time =
//           compressor_->decompress(D_, shape_, tol, comp_data, comp_size,
//                                   decompressed, compIdx == nComponents - 1);

//       compressTimes_[compIdx] = comp_time + decomp_time;

//       T *decompressed_data = static_cast<T *>(decompressed);
//       T max_xi = 0.0;
//       for (size_t j = 0; j < n_; j++) {
//         max_xi = std::max(max_xi, std::fabs(decompressed_data[j]));
//       }
//       std::cout << "[Comp " << compIdx + 1 << "] Max |x_i| = " << max_xi
//                 << std::endl;

//       for (size_t j = 0; j < n_; j++) {
//         xTilde_[j] += decompressed_data[j];
//       }

//       T maxErr = 0.0;
//       for (size_t j = 0; j < n_; j++) {
//         T diff = std::fabs(original_data[j] - xTilde_[j]);
//         if (diff > maxErr)
//           maxErr = diff;
//       }
//       std::cout << "Component " << compIdx + 1 << ": max error = " << maxErr
//                 << ", tol = " << tol << std::endl;

//       delete[] decompressed_data;
//     }

//     return true;
//   }

//   T *reconstructData(int m) {
//     if (m > num_components_)
//       m = num_components_;
//     T *result = new T[n_];
//     std::fill(result, result + n_, 0.0);

//     for (int i = 0; i < m; i++) {
//       void *decompressed = nullptr;
//       double tol = components_[i].tol;
//       double ok = compressor_->decompress(
//           D_, shape_, tol, components_[i].compressed_data,
//           components_[i].compressed_size, decompressed, false);
//       if (!decompressed) {
//         std::cerr << "[Warning] decompress() returned nullptr at component "
//                   << (i + 1) << std::endl;
//         continue;
//       }
//       T *decompressed_data = static_cast<T *>(decompressed);
// #if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR && \
//     defined(__CUDACC__)
//       launchAccumulateKernel<T>(result, decompressed_data, n_);
// #else
//       launchAccumulateCPU<T>(result, decompressed_data, n_);
// #endif
//       delete[] decompressed_data;
//     }

//     return result;
//   }

//   T *reconstructData(double reqTol) {
//     int m = requestComponentsForBound(reqTol);
//     return reconstructData(m);
//   }

//   int getNumComponents() const { return num_components_; }

//   size_t getCompressedSize(int componentIdx) const {
//     if (componentIdx < 0 || componentIdx >= num_components_)
//       return 0;
//     return components_[componentIdx].compressed_size;
//   }

//   double getComponentCompressTime(int idx) const {
//     if (idx < 0 || idx >= compressTimes_.size())
//       return 0;
//     return compressTimes_[idx];
//   }

//   int requestComponentsForBound(double reqBound) {
//     int requiredComponents = 0;
//     for (int i = 0; i < num_components_; ++i) {
//       if (components_[i].tol <= reqBound) {
//         requiredComponents = i + 1;
//         break;
//       }
//     }
//     if (requiredComponents == 0)
//       requiredComponents = num_components_;
//     return requiredComponents;
//   }

// private:
//   GeneralCompressor<T> *compressor_;
//   std::vector<Component> components_;
//   std::vector<double> compressTimes_;
//   int num_components_;
//   T *xTilde_;
//   size_t n_;
//   mgard_x::DIM D_;
//   std::vector<mgard_x::SIZE> shape_;
// };

// #endif // PROGRESSIVE_COMPRESSOR_HPP

#ifndef PROGRESSIVE_COMPRESSOR_HPP
#define PROGRESSIVE_COMPRESSOR_HPP

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

#ifdef ENABLE_CUDA_COMPRESSOR
#include <cuda_runtime.h>
#else
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR &&               \
    defined(__CUDACC__)
template <typename T>
__global__ void computeErrorKernel(const T *d_orig, const T *d_xTilde,
                                   T *d_error, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_error[idx] = d_orig[idx] - d_xTilde[idx];
  }
}

template <typename T>
void launchComputeErrorKernel(const T *h_orig, T *h_xTilde, T *h_error,
                              size_t n) {
  T *d_orig = nullptr, *d_xTilde = nullptr, *d_error = nullptr;
  cudaMalloc(&d_orig, n * sizeof(T));
  cudaMalloc(&d_xTilde, n * sizeof(T));
  cudaMalloc(&d_error, n * sizeof(T));

  cudaMemcpy(d_orig, h_orig, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xTilde, h_xTilde, n * sizeof(T), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  computeErrorKernel<T><<<blocks, threads>>>(d_orig, d_xTilde, d_error, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_error, d_error, n * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_orig);
  cudaFree(d_xTilde);
  cudaFree(d_error);
}

template <typename T>
__global__ void accumulateKernel(const T *d_decompressed, T *d_result,
                                 size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_result[idx] += d_decompressed[idx];
  }
}

template <typename T>
void launchAccumulateKernel(T *h_result, T *h_decompressed, size_t n) {
  T *d_result = nullptr, *d_decompressed = nullptr;
  cudaMalloc(&d_result, n * sizeof(T));
  cudaMalloc(&d_decompressed, n * sizeof(T));

  cudaMemcpy(d_result, h_result, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_decompressed, h_decompressed, n * sizeof(T),
             cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  accumulateKernel<T><<<blocks, threads>>>(d_decompressed, d_result, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_result, d_result, n * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_result);
  cudaFree(d_decompressed);
}

#else
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
#endif

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

  bool compressData(const T *original_data, const double *toleranceList,
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

    for (int compIdx = 0; compIdx < nComponents; compIdx++) {
      double tol = toleranceList[compIdx];
      T *error = new T[n_];

#if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR &&               \
    defined(__CUDACC__)
      launchComputeErrorKernel<T>(original_data, xTilde_, error, n_);
#else
      launchComputeErrorCPU<T>(original_data, xTilde_, error, n_);
#endif

      T max_residual = 0.0;
      for (size_t i = 0; i < n_; i++) {
        max_residual = std::max(max_residual, std::fabs(error[i]));
      }
      std::cout << "[Comp " << compIdx + 1
                << "] Residual max = " << max_residual << ", tol = " << tol
                << std::endl;

      void *comp_data = nullptr;
      size_t comp_size = 0;
      double comp_time = compressor_->compress(
          D_, shape_, tol, error, comp_data, comp_size, compIdx == 0);

      delete[] error;

      components_[compIdx].compressed_data = comp_data;
      components_[compIdx].compressed_size = comp_size;
      components_[compIdx].tol = tol;

      void *decompressed = nullptr;
      double decomp_time =
          compressor_->decompress(D_, shape_, tol, comp_data, comp_size,
                                  decompressed, compIdx == nComponents - 1);

      compressTimes_[compIdx] = comp_time + decomp_time;

      T *decompressed_data = static_cast<T *>(decompressed);
      T max_xi = 0.0;
      for (size_t j = 0; j < n_; j++) {
        max_xi = std::max(max_xi, std::fabs(decompressed_data[j]));
      }
      std::cout << "[Comp " << compIdx + 1 << "] Max |x_i| = " << max_xi
                << std::endl;

      for (size_t j = 0; j < n_; j++) {
        xTilde_[j] += decompressed_data[j];
      }

      T maxErr = 0.0;
      for (size_t j = 0; j < n_; j++) {
        T diff = std::fabs(original_data[j] - xTilde_[j]);
        if (diff > maxErr)
          maxErr = diff;
      }
      std::cout << "Component " << compIdx + 1 << ": max error = " << maxErr
                << ", tol = " << tol << std::endl;

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
      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();
      component_decomp_times_[i] = dt;

      if (!decompressed) {
        std::cerr << "[Warning] decompress() returned nullptr at component "
                  << (i + 1) << std::endl;
        continue;
      }
      T *decompressed_data = static_cast<T *>(decompressed);
#if defined(ENABLE_CUDA_COMPRESSOR) && ENABLE_CUDA_COMPRESSOR &&               \
    defined(__CUDACC__)
      launchAccumulateKernel<T>(result, decompressed_data, n_);
#else
      launchAccumulateCPU<T>(result, decompressed_data, n_);
#endif
      delete[] decompressed_data;

      std::cout << "Component " << i + 1
                << ": incremental decompression time = " << dt << " s"
                << std::endl;
    }
    return result;
  }

  
  const std::vector<double> &getComponentDecompressionTimes() const {
    return component_decomp_times_;
  }

  T *reconstructData(double reqTol) {
    int m = requestComponentsForBound(reqTol);
    return reconstructData(m);
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

  int requestComponentsForBound(double reqBound) {
    int requiredComponents = 0;
    for (int i = 0; i < num_components_; ++i) {
      if (components_[i].tol <= reqBound) {
        requiredComponents = i + 1;
        break;
      }
    }
    if (requiredComponents == 0)
      requiredComponents = num_components_;
    return requiredComponents;
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

#endif // PROGRESSIVE_COMPRESSOR_HPP
