#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CPUSZCompressor.hpp"
#include "CPUZFPCompressor.hpp"
#include "ProgressiveCompressorCPU.hpp"

using namespace std;

void print_usage_message(const std::string &error) {
  if (!error.empty()) {
    std::cerr << "Error: " << error << "\n";
  }
  std::cout
      << "Usage:\n"
         "  -i <input file>                       : dataset file path\n"
         "  -n <ndim> <dim1> <dim2> ... <dimN>    : number of dimensions and "
         "dims (space separated)\n"
         "  -ECount <error count>                 : number of errors\n"
         "  -E <error count> <error1> <error2> ... <errorN>     : error list "
         "(space "
         "separated)\n"
         "  -p <s|d>                              : precision (s: single, d: "
         "double)\n"
         "  -help                                 : display this message\n";
  exit(1);
}

bool has_arg(int argc, char *argv[], const char *option) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], option) == 0) {
      return true;
    }
  }
  return false;
}

std::string get_arg(int argc, char *argv[], const char *option) {
  for (int i = 0; i < argc - 1; i++) {
    if (strcmp(argv[i], option) == 0) {
      return std::string(argv[i + 1]);
    }
  }
  print_usage_message(std::string("Missing required option: ") + option);
  return "";
}

int get_arg_int(int argc, char *argv[], const char *option) {
  std::string arg = get_arg(argc, argv, option);
  try {
    return std::stoi(arg);
  } catch (...) {
    print_usage_message(std::string("Invalid integer for option: ") + option);
  }
  return 0;
}

double get_arg_double(int argc, char *argv[], const char *option) {
  std::string arg = get_arg(argc, argv, option);
  try {
    return std::stod(arg);
  } catch (...) {
    print_usage_message(std::string("Invalid double for option: ") + option);
  }
  return 0;
}

template <typename T>
std::vector<T> get_arg_dims(int argc, char *argv[], const char *option) {
  std::vector<T> vec;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], option) == 0) {
      if (i + 1 >= argc) {
        print_usage_message(std::string("Missing count after option: ") +
                            option);
      }
      int count = std::stoi(argv[i + 1]);
      for (int j = 0; j < count; j++) {
        if (i + 2 + j >= argc) {
          print_usage_message(std::string("Missing value for option: ") +
                              option);
        }
        T val;
        std::istringstream iss(argv[i + 2 + j]);
        iss >> val;
        vec.push_back(val);
      }
      return vec;
    }
  }
  print_usage_message(std::string("Missing required option: ") + option);
  return vec;
}

template <typename T>
T *readFile(const std::string &filename, size_t &numElements) {
  std::ifstream infile(filename, std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    numElements = 0;
    return nullptr;
  }
  std::streamsize fileSize = infile.tellg();
  infile.seekg(0, std::ios::beg);
  if (fileSize % sizeof(T) != 0) {
    std::cerr << "File size is not a multiple of data type size." << std::endl;
    numElements = 0;
    return nullptr;
  }
  numElements = fileSize / sizeof(T);
  T *data = new T[numElements];
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
                         const std::vector<double> &absTolList, std::string filename) {
  std::cout << "\n=== Testing " << name << " ===\n";
  ProgressiveCompressor<T> prog(D, shape, compressor);
  bool ok = prog.compressData(data, absTolList.data(), absTolList.size());
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
  std::ofstream out("result_" + name + ".csv", std::ios::app);
  out << "Results for file: " << filename << "\n";
  out << "Index,Tolerance,Time(s),CompressedSize(Bytes)\n";
  for (int i = 0; i < prog.getNumComponents(); i++) {
    out << i + 1 << "," << absTolList[i] << ","
        << prog.getComponentCompressTime(i) << "," << prog.getCompressedSize(i)
        << "\n";
  }
  double refactorTime = 0;
  for (int i = 0; i < prog.getNumComponents(); i++) {
    refactorTime += prog.getComponentCompressTime(i);
  }
  out << "Refactor Time " << refactorTime << "\n";
  out.close();
}

template <typename T>
void testFullProgressiveReconstruct(const std::string &name,
                                    GeneralCompressor<T> *compressor,
                                    mgard_x::DIM D,
                                    const std::vector<mgard_x::SIZE> &shape,
                                    T *data,
                                    const std::vector<double> &relTolList,
                                  std::string filename) {
  std::cout << "\n=== Testing Reconstruction for Error Bounds (" << name
            << ") ===\n";
  ProgressiveCompressor<T> prog(D, shape, compressor);
  bool ok = prog.compressData(data, relTolList.data(), relTolList.size());
  if (!ok) {
    std::cerr << "[FAIL] Progressive compression failed." << std::endl;
    return;
  }
  std::ofstream out("result_reconstruct_bound_" + name + ".csv", std::ios::app);
  out << "Results for file: " << filename << "\n";
  out << "Component,TargetTolerance,DecompressionTime(s)\n";

  T *reconstructed = prog.reconstructData(relTolList.size());
  if (reconstructed == nullptr) {
    std::cerr << "[FAIL] Reconstruction failed for target tolerance"
              << std::endl;
    exit(-1);
  }
  delete[] reconstructed;
  const std::vector<double> &compTimes = prog.getComponentDecompressionTimes();
  for (size_t comp = 0; comp < compTimes.size(); comp++) {
    out << comp + 1 << "," << relTolList[comp] << "," << compTimes[comp]
        << "\n";
  }

  out.close();
}

template <typename T>
int run_test(int argc, char *argv[]) {
  std::string filename = get_arg(argc, argv, "-i");

  std::vector<mgard_x::SIZE> shape =
      get_arg_dims<mgard_x::SIZE>(argc, argv, "-n");
  mgard_x::DIM D = shape.size();

  int errorCount = get_arg_int(argc, argv, "-ECount");

  std::vector<double> relTolList = get_arg_dims<double>(argc, argv, "-E");
  if ((int)relTolList.size() != errorCount) {
    std::cerr << "Error: error count does not match number of error values "
                 "provided.\n";
    exit(1);
  }

  size_t numElements = 0;
  T *data = readFile<T>(filename, numElements);
  if (data == nullptr) {
    std::cerr << "Failed to read dataset.\n";
    exit(1);
  }

  std::vector<std::pair<std::string, GeneralCompressor<T> *>> compressors;

  CPUSZCompressor<T> sz_cpu;
  CPUZFPCompressor<T> zfp_cpu;

  compressors.push_back(std::make_pair("SZ3_CPU", &sz_cpu));
  compressors.push_back(std::make_pair("ZFP_CPU", &zfp_cpu));

  for (auto &p : compressors) {
    std::cout << "\n========================================\n";
    std::cout << "Testing compressor: " << p.first << "\n";
    testProgressiveComp<T>(p.first, p.second, D, shape, data, relTolList, filename);
    testFullProgressiveReconstruct<T>(p.first, p.second, D, shape, data,
                                      relTolList,filename);
  }
  delete[] data;
  return 0;
}

int main(int argc, char *argv[]) {
  if (has_arg(argc, argv, "-help")) {
    print_usage_message("");
  }

  std::string prec = "s";
  if (has_arg(argc, argv, "-p")) {
    prec = get_arg(argc, argv, "-p");
  }
  if (prec == "d") {
    return run_test<double>(argc, argv);
  } else if (prec == "s") {
    return run_test<float>(argc, argv);
  } else {
    print_usage_message("Unknown precision option: " + prec);
  }
  return 0;
}
