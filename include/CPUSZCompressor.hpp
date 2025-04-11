#ifndef CPU_SZ_COMPRESSOR_HPP
#define CPU_SZ_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "SZ3/api/sz.hpp"
// #include <omp.h>
// omp_set_num_threads(24);

// OPENMP

template <typename T>
class CPUSZCompressor : public GeneralCompressor<T>
{
public:
    bool compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size) override
    {
        SZ::Config conf;
        if (D == 1)
        {
            conf = SZ::Config(shape[0]);
        }
        else if (D == 2)
        {
            conf = SZ::Config(shape[0], shape[1]);
        }
        else if (D == 3)
        {
            conf = SZ::Config(shape[0], shape[1], shape[2]);
        }
        else
        {
            std::cout << "wrong D\n";
            exit(-1);
        }

        conf.errorBoundMode = SZ::EB_ABS;
        conf.absErrorBound = tol;
        conf.openmp = true;

        char *cmp_data = SZ_compress(conf, original_data, compressed_size);
        compressed_data = malloc(compressed_size);
        memcpy(compressed_data, cmp_data, compressed_size);
        delete[] cmp_data;

        return true;
    }

    bool decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                    void *compressed_data, size_t compressed_size,
                    void *&decompressed_data) override
    {
        SZ::Config conf;
        size_t total_elems = 1;
        if (D == 1)
        {
            conf = SZ::Config(shape[0]);
            total_elems = shape[0];
        }
        else if (D == 2)
        {
            conf = SZ::Config(shape[0], shape[1]);
            total_elems = shape[0] * shape[1];
        }
        else if (D == 3)
        {
            conf = SZ::Config(shape[0], shape[1], shape[2]);
            total_elems = shape[0] * shape[1] * shape[2];
        }
        else
        {
            std::cout << "wrong D\n";
            exit(-1);
        }

        conf.errorBoundMode = SZ::EB_ABS;
        conf.absErrorBound = tol;
        conf.openmp = true;

        decompressed_data = malloc(sizeof(T) * total_elems);

        T *dec_data = (T *)decompressed_data;
        SZ_decompress(conf, (char *)compressed_data, compressed_size, dec_data);

        return true;
    }
};

#endif
// CPU_SZ_COMPRESSOR_HPP
