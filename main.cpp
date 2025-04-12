#include "CPUSZCompressor.hpp"
#include "CPUZFPCompressor.hpp"
#include "ProgressiveCompressor.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>

#if ENABLE_CUDA_COMPRESSOR
#include "GPUMGARDCompressor.hpp"
#include "GPUSZCompressor.hpp"
#include "GPUZFPCompressor.hpp"
#endif

using namespace std;

// Modify when using different precision
using PRECISION = float;

PRECISION *readFile(const std::string &filename, size_t &numElements) {
  std::ifstream infile(filename, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    numElements = 0;
    return nullptr;
  }
  std::streamsize fileSize = infile.tellg();
  infile.seekg(0, std::ios::beg);

  if (fileSize % sizeof(PRECISION) != 0) {
    std::cerr << "File size is not a multiple of PRECISION size." << std::endl;
    numElements = 0;
    return nullptr;
  }

  numElements = fileSize / sizeof(PRECISION);
  PRECISION *data = new PRECISION[numElements];
  if (!infile.read(reinterpret_cast<char *>(data), fileSize)) {
    std::cerr << "Error reading file: " << filename << std::endl;
    delete[] data;
    numElements = 0;
    return nullptr;
  }
  return data;
}

template <typename T>
void testProgressiveComp(const std::string &name,
                         GeneralCompressor<T> *compressor, mgard_x::DIM D,
                         const std::vector<mgard_x::SIZE> &shape, T *data,
                         const std::vector<double> &absTolList) {
  std::cout << "\n=== Testing " << name << " ===\n";

  ProgressiveCompressor<T> prog(D, shape, compressor);

  auto t0 = std::chrono::high_resolution_clock::now();
  bool ok = prog.compressData(data, absTolList.data(), absTolList.size());
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!ok) {
    std::cerr << "[FAIL] Progressive compression failed." << std::endl;
    return;
  }

  for (int i = 0; i < prog.getNumComponents(); ++i) {
    double compTime = prog.getComponentCompressTime(i);
    size_t compSize = prog.getCompressedSize(i);
    std::cout << "Component " << i + 1 << ": tol = " << absTolList[i]
              << ", compress time = " << compTime << " s"
              << ", size = " << compSize << " bytes" << std::endl;
  }

  std::ofstream out("result_" + name + ".csv");
  out << "Index,Tolerance,Time(s),CompressedSize(Bytes)\n";
  for (int i = 0; i < prog.getNumComponents(); i++) {
    out << i + 1 << "," << absTolList[i] << ","
        << prog.getComponentCompressTime(i) << "," << prog.getCompressedSize(i)
        << "\n";
  }
  out.close();
}

template <typename T>
void testProgressiveReconstructForBound(const std::string &name,
                                        GeneralCompressor<T> *compressor,
                                        mgard_x::DIM D,
                                        const std::vector<mgard_x::SIZE> &shape,
                                        T *data,
                                        const std::vector<double> &absTolList) {
  std::cout << "\n=== Testing Reconstruction for Error Bounds (" << name
            << ") ===\n";

  ProgressiveCompressor<T> prog(D, shape, compressor);
  bool ok = prog.compressData(data, absTolList.data(), absTolList.size());
  if (!ok) {
    std::cerr << "[FAIL] Progressive compression failed." << std::endl;
    return;
  }

  std::ofstream out("result_reconstruct_bound_" + name + ".csv");
  out << "Index,TargetTolerance,RequiredComponents,ReconstructTime(s)\n";

  for (int i = 0; i < absTolList.size(); i++) {
    double targetTol = absTolList[i];
    int reqComps = prog.requestComponentsForBound(targetTol);
    auto t0 = std::chrono::high_resolution_clock::now();
    PRECISION *reconstructed = prog.reconstructData(reqComps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double recTime = std::chrono::duration<double>(t1 - t0).count();

    if (reconstructed == nullptr) {
      std::cerr << "[FAIL] Reconstruction failed for target tolerance "
                << targetTol << std::endl;
    } else {
      std::cout << "Target tolerance " << targetTol
                << ": required components = " << reqComps
                << ", reconstruction time = " << recTime << " s" << std::endl;
      delete[] reconstructed;
    }
    out << i + 1 << "," << targetTol << "," << reqComps << "," << recTime
        << "\n";
  }
  out.close();
}

int main() {
  omp_set_num_threads(64);
  std::string filename = "/home/leonli/SDRBENCH/single_precision/"
                         "SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32";
  size_t numElements = 512 * 512 * 512;
  PRECISION *data = readFile(filename, numElements);
  if (data == nullptr) {
    std::cerr << "Failed to read file." << std::endl;
    return 1;
  }

  mgard_x::DIM D = 3;
  std::vector<mgard_x::SIZE> shape = {512, 512, 512};
  const int nComponents = 11;
  std::vector<double> relTolList = {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4,
                                    1e-4, 5e-5, 1e-5, 5e-6, 1e-6};

  // Look up on SDRBENCH for official dataset property for temperature.f32
  double valueRange = 4780302.524170;

  std::vector<double> absTolList(nComponents);
  for (int i = 0; i < nComponents; i++) {
    absTolList[i] = valueRange * relTolList[i];
  }

#if ENABLE_CUDA_COMPRESSOR
  GPUMGARDCompressor<PRECISION> mgard;
  GPUZFPCompressor<PRECISION> zfp_gpu;
  GPUSZCompressor<PRECISION> cuszp;
#endif
  CPUSZCompressor<PRECISION> sz_cpu;
  CPUZFPCompressor<PRECISION> zfp_cpu;

  std::vector<std::pair<std::string, GeneralCompressor<PRECISION> *>>
      compressors;

#if ENABLE_CUDA_COMPRESSOR
  compressors.emplace_back("MGARD", &mgard);
  compressors.emplace_back("ZFP_GPU", &zfp_gpu);
  compressors.emplace_back("cuSZp", &cuszp);
#endif
  compressors.emplace_back("SZ3_CPU", &sz_cpu);
  compressors.emplace_back("ZFP_CPU", &zfp_cpu);

  for (auto &[name, comp] : compressors) {
    // testProgressiveComp(name, comp, D, shape, data, absTolList);
    testProgressiveReconstructForBound(name, comp, D, shape, data, absTolList);
  }

  delete[] data;
  return 0;
}
