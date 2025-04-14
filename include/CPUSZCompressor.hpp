#ifndef CPU_SZ_COMPRESSOR_HPP
#define CPU_SZ_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "SZ3/api/sz.hpp"
// #include <omp.h>
// omp_set_num_threads(24);

// OPENMP

template <typename T> class CPUSZCompressor : public GeneralCompressor<T> {
public:
  double compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size, bool include_copy_time) override {
    auto t0 = std::chrono::high_resolution_clock::now();
    SZ::Config conf;
    if (D == 1) {
      conf = SZ::Config(shape[0]);
    } else if (D == 2) {
      conf = SZ::Config(shape[0], shape[1]);
    } else if (D == 3) {
      conf = SZ::Config(shape[0], shape[1], shape[2]);
    } else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    conf.errorBoundMode = SZ::EB_ABS;
    conf.absErrorBound = tol;
    conf.openmp = true;

    auto t1 = std::chrono::high_resolution_clock::now();

    char *cmp_data = SZ_compress(conf, original_data, compressed_size);

    auto t2 = std::chrono::high_resolution_clock::now();

    compressed_data = malloc(compressed_size);
    memcpy(compressed_data, cmp_data, compressed_size);
    delete[] cmp_data;

    double elapsed_time;
    if (include_copy_time) {
      elapsed_time = std::chrono::duration<double>(t2 - t0).count();
    } else {
      elapsed_time = std::chrono::duration<double>(t2 - t1).count();
    }

    return elapsed_time;
  }

  double decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape,
                    double tol, void *compressed_data, size_t compressed_size,
                    void *&decompressed_data, bool include_copy_time) override {
    SZ::Config conf;
    size_t total_elems = 1;
    if (D == 1) {
      conf = SZ::Config(shape[0]);
      total_elems = shape[0];
    } else if (D == 2) {
      conf = SZ::Config(shape[0], shape[1]);
      total_elems = shape[0] * shape[1];
    } else if (D == 3) {
      conf = SZ::Config(shape[0], shape[1], shape[2]);
      total_elems = shape[0] * shape[1] * shape[2];
    } else {
      std::cout << "wrong D\n";
      exit(-1);
    }

    conf.errorBoundMode = SZ::EB_ABS;
    conf.absErrorBound = tol;
    conf.openmp = true;

    decompressed_data = malloc(sizeof(T) * total_elems);

    T *dec_data = (T *)decompressed_data;

    auto t0 = std::chrono::high_resolution_clock::now();

    SZ_decompress(conf, (char *)compressed_data, compressed_size, dec_data);

    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

    return elapsed_time;
  }
};

#endif
// CPU_SZ_COMPRESSOR_HPP
