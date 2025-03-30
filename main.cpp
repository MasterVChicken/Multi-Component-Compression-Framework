#include "GPUMGARDCompressor.hpp"
#include "ProgressiveCompressor.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

float *readF32File(const std::string &filename, size_t &numElements) {
  // 以二进制模式打开文件，并定位到文件末尾以获取文件大小
  std::ifstream infile(filename, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    numElements = 0;
    return nullptr;
  }
  std::streamsize fileSize = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // 检查文件大小是否为 float 大小的整数倍
  if (fileSize % sizeof(float) != 0) {
    std::cerr << "File size is not a multiple of float size." << std::endl;
    numElements = 0;
    return nullptr;
  }

  numElements = fileSize / sizeof(float);
  float *data = new float[numElements];
  if (!infile.read(reinterpret_cast<char *>(data), fileSize)) {
    std::cerr << "Error reading file: " << filename << std::endl;
    delete[] data;
    numElements = 0;
    return nullptr;
  }
  return data;
}

double *readF64File(const std::string &filename, size_t &numElements) {
  // 以二进制方式打开文件，并定位到文件末尾以获取文件大小
  std::ifstream infile(filename, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    numElements = 0;
    return nullptr;
  }
  std::streamsize fileSize = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // 检查文件大小是否为 double 大小的整数倍
  if (fileSize % sizeof(double) != 0) {
    std::cerr << "File size is not a multiple of double size." << std::endl;
    numElements = 0;
    return nullptr;
  }

  numElements = fileSize / sizeof(double);
  double *data = new double[numElements];
  if (!infile.read(reinterpret_cast<char *>(data), fileSize)) {
    std::cerr << "Error reading file: " << filename << std::endl;
    delete[] data;
    numElements = 0;
    return nullptr;
  }
  return data;
}

int main() {
  // Using random dataset with 1000 elements
  // const int n = 1000;
  // double *data = new double[n];

  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<double> dis(-10.0, 10.0);
  // for (int i = 0; i < n; i++) {
  //   data[i] = dis(gen);
  // }

  std::string filename =
      "/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/"
      "density.d64";
  size_t numElements = 384 * 384 * 256;
  double *data = readF64File(filename, numElements);
  if (data == nullptr) {
    std::cerr << "Failed to read file." << std::endl;
    return 1;
  }

  int shape[3] = {384, 384, 256};

  const int nComponents = 10;
  double toleranceList[nComponents] = {5,   2.5, 1.25, 0.625, 0.3225,
                                       0.2, 0.1, 0.05, 0.01,  0.001};

  GPUMGARDCompressor mgardCompressor;
  ProgressiveCompressor progComp(&mgardCompressor);

  bool ok = progComp.compressData(3, shape, DataType::DOUBLE, data,
                                  toleranceList, nComponents);
  if (!ok) {
    cerr << "Progressive compression failed." << endl;
    delete[] data;
    return 1;
  }

  // Use some components to reconstruct data
  double *reconstructed = progComp.reconstructData(nComponents - 3);
  // double *reconstructed = progComp.reconstructData(1);
  if (!reconstructed) {
    cerr << "Progressive reconstruction failed." << endl;
    delete[] data;
    return 1;
  }

  cout << "Index\tOriginal\tReconstructed" << endl;
  for (int i = 10000; i < 10020; i++) {
    cout << i << "\t" << data[i] << "\t" << reconstructed[i] << endl;
  }

  delete[] data;
  delete[] reconstructed;

  return 0;
}
