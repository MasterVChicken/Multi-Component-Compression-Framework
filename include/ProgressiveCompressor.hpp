#ifndef PROGRESSIVE_COMPRESSOR_HPP
#define PROGRESSIVE_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

template <typename T>
__global__ void computeErrorKernel(const T *d_orig, const T *d_xTilde, T *d_error, size_t n)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    d_error[idx] = d_orig[idx] - d_xTilde[idx];
  }
}

template <typename T>
void launchComputeErrorKernel(const T *h_orig, T *h_xTilde, T *h_error, size_t n)
{
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
__global__ void accumulateKernel(const T *d_decompressed, T *d_result, size_t n)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    d_result[idx] += d_decompressed[idx];
  }
}

template <typename T>
void launchAccumulateKernel(T *h_result, T *h_decompressed, size_t n)
{
  T *d_result = nullptr, *d_decompressed = nullptr;
  cudaMalloc(&d_result, n * sizeof(T));
  cudaMalloc(&d_decompressed, n * sizeof(T));

  cudaMemcpy(d_result, h_result, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_decompressed, h_decompressed, n * sizeof(T), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  accumulateKernel<T><<<blocks, threads>>>(d_decompressed, d_result, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_result, d_result, n * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_result);
  cudaFree(d_decompressed);
}

template <typename T>
class ProgressiveCompressor
{
public:
  struct Component
  {
    void *compressed_data;
    size_t compressed_size;
    double tol;
  };

  ProgressiveCompressor(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                        GeneralCompressor<T> *compressor)
      : D_(D), shape_(shape), compressor_(compressor),
        num_components_(0), xTilde_(nullptr), n_(0) {}

  ~ProgressiveCompressor()
  {
    for (int i = 0; i < num_components_; i++)
    {
      if (components_[i].compressed_data != nullptr)
      {
        free(components_[i].compressed_data);
      }
    }
    if (xTilde_)
    {
      delete[] xTilde_;
    }
  }

  bool compressData(const T *original_data, const double *toleranceList, int nComponents)
  {
    n_ = 1;
    for (int i = 0; i < D_; i++)
    {
      n_ *= shape_[i];
    }

    xTilde_ = new T[n_];
    for (size_t i = 0; i < n_; i++)
    {
      xTilde_[i] = 0.0;
    }

    components_.resize(nComponents);
    compressTimes_.resize(nComponents, 0.0);
    num_components_ = nComponents;

    for (int compIdx = 0; compIdx < nComponents; compIdx++)
    {
      double tol = toleranceList[compIdx];

      auto comp_t0 = std::chrono::high_resolution_clock::now();

      T *error = new T[n_];
      launchComputeErrorKernel<T>(original_data, xTilde_, error, n_);

      void *comp_data = nullptr;
      size_t comp_size = 0;
      bool ok = compressor_->compress(D_, shape_, tol, error, comp_data, comp_size);
      delete[] error;
      if (!ok)
      {
        std::cerr << "Compression failed at component " << compIdx + 1 << std::endl;
        return false;
      }
      components_[compIdx].compressed_data = comp_data;
      components_[compIdx].compressed_size = comp_size;
      components_[compIdx].tol = tol;

      auto comp_t1 = std::chrono::high_resolution_clock::now();
      compressTimes_[compIdx] = std::chrono::duration<double>(comp_t1 - comp_t0).count();

      void *decompressed = nullptr;
      ok = compressor_->decompress(D_, shape_, tol, comp_data, comp_size, decompressed);
      if (!ok || decompressed == nullptr)
      {
        std::cerr << "[ERROR] Decompression failed at component " << compIdx + 1 << std::endl;
        std::cerr << "[ERROR] comp_size = " << comp_size << std::endl;
        std::cerr << "[ERROR] comp_data ptr = " << comp_data << std::endl;
        return false;
      }
      T *decompressed_data = static_cast<T *>(decompressed);
      T maxErr = 0.0;
      for (size_t j = 0; j < n_; j++)
      {
        xTilde_[j] += decompressed_data[j];
        T diff = std::fabs(original_data[j] - xTilde_[j]);
        if (diff > maxErr)
          maxErr = diff;
      }
      delete[] decompressed_data;
      std::cout << "Component " << compIdx + 1 << ": max error = " << maxErr
                << ", tol = " << tol << std::endl;
    }

    if (!writeMetadata("metadata.txt", D_, shape_))
    {
      std::cerr << "Failed to write metadata file." << std::endl;
      return false;
    }
    return true;
  }

  T *reconstructData(int m)
  {
    if (m > num_components_)
      m = num_components_;
    T *result = new T[n_];
    for (size_t j = 0; j < n_; j++)
    {
      result[j] = 0.0;
    }
    for (int i = 0; i < m; i++)
    {
      std::cout << "Component " << i + 1 << " has been used!" << std::endl;
      void *decompressed = nullptr;

      double dummyTol = 0;
      bool ok = compressor_->decompress(D_, shape_, dummyTol, components_[i].compressed_data,
                                        components_[i].compressed_size, decompressed);
      if (!ok || decompressed == nullptr)
      {
        std::cerr << "Decompression failed in reconstructData at component " << i + 1 << std::endl;
        delete[] result;
        return nullptr;
      }
      T *decompressed_data = static_cast<T *>(decompressed);
      launchAccumulateKernel<T>(result, decompressed_data, n_);
      delete[] decompressed_data;
    }
    return result;
  }

  int getNumComponents() const { return num_components_; }

  size_t getCompressedSize(int componentIdx) const
  {
    if (componentIdx < 0 || componentIdx >= num_components_)
      return 0;
    return components_[componentIdx].compressed_size;
  }

  double getComponentCompressTime(int idx) const
  {
    if (idx < 0 || idx >= compressTimes_.size())
      return 0;
    return compressTimes_[idx];
  }

  int requestComponentsForBound(double reqBound)
  {
    std::ifstream ifs("metadata.txt");
    if (!ifs.is_open())
    {
      std::cerr << "Failed to open metadata file: metadata.txt" << std::endl;
      return 0;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
      if (line.empty() || line[0] == '#')
        continue;
      if (line.find("ComponentIndex") != std::string::npos)
        break;
    }
    int componentCount = 0;
    int requiredComponents = 0;
    while (std::getline(ifs, line))
    {
      if (line.empty())
        continue;
      std::istringstream iss(line);
      int index;
      double tol;
      size_t compSize;
      if (!(iss >> index >> tol >> compSize))
        continue;
      componentCount++;
      if (tol <= reqBound)
      {
        requiredComponents = index;
        break;
      }
    }
    if (requiredComponents == 0)
      requiredComponents = componentCount;
    return requiredComponents;
  }

  bool writeMetadata(const std::string &filename, int D, const std::vector<mgard_x::SIZE> &shape)
  {
    std::ofstream ofs(filename.c_str());
    if (!ofs.is_open())
    {
      std::cerr << "Failed to open metadata file: " << filename << std::endl;
      return false;
    }
    ofs << "# Progressive Compressor Metadata File" << std::endl;
    ofs << "Dimensions: " << D << std::endl;
    ofs << "Shape:";
    for (auto s : shape)
    {
      ofs << " " << s;
    }
    ofs << std::endl;
    ofs << "nComponents: " << num_components_ << std::endl;
    ofs << "ComponentIndex Tolerance CompressedSize" << std::endl;
    for (int i = 0; i < num_components_; i++)
    {
      ofs << (i + 1) << " " << components_[i].tol << " " << components_[i].compressed_size << std::endl;
    }
    ofs.close();
    return true;
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
};

#endif // PROGRESSIVE_COMPRESSOR_HPP
