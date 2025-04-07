#ifndef GPU_SZ_COMPRESSOR_HPP
#define GPU_SZ_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include <cuSZp.h>
#include <cuda_runtime.h>
// #include <omp.h>
// omp_set_num_threads(24);

template <typename T>
class GPUSZCompressor : public GeneralCompressor<T>
{
public:
    bool compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size) override
    {
        cuszp_type_t dataType;
        // or we can use CUSZP_MODE_OUTLIER
        cuszp_mode_t encodingMode = CUSZP_MODE_PLAIN;

        if (std::is_same<T, double>::value)
        {
            dataType = CUSZP_TYPE_DOUBLE;
        }
        else if (std::is_same<T, float>::value)
        {
            dataType = CUSZP_TYPE_FLOAT;
        }
        else
        {
            std::cout << "wrong dtype\n";
            exit(-1);
        }

        size_t data_size = 1;
        for (int i = 0; i < D; i++)
        {
            data_size *= shape[i];
        }

        T *d_oriData;
        unsigned char *d_cmpBytes;
        cudaMalloc((void **)&d_oriData, data_size * sizeof(T));
        cudaMemcpy(d_oriData, original_data, sizeof(T) * data_size,
                   cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_cmpBytes, data_size * sizeof(T));

        // Initializing CUDA Stream.
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // d_oriData and d_cmpBytes are device pointers
        cuSZp_compress(d_oriData, d_cmpBytes, data_size, &compressed_size, tol,
                       dataType, encodingMode, stream);

        compressed_data = malloc(compressed_size);
        cudaMemcpy(compressed_data, d_cmpBytes, compressed_size,
                   cudaMemcpyDeviceToHost);
        cudaFree(d_oriData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);

        return true;
    }

    bool decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                    void *compressed_data, size_t compressed_size,
                    void *&decompressed_data) override
    {
        cuszp_type_t dataType;
        // or we can use CUSZP_MODE_OUTLIER
        cuszp_mode_t encodingMode = CUSZP_MODE_PLAIN;

        if (std::is_same<T, double>::value)
        {
            dataType = CUSZP_TYPE_DOUBLE;
        }
        else if (std::is_same<T, float>::value)
        {
            dataType = CUSZP_TYPE_FLOAT;
        }
        else
        {
            std::cout << "wrong dtype\n";
            exit(-1);
        }

        size_t data_size = 1;
        for (int i = 0; i < D; i++)
        {
            data_size *= shape[i];
        }

        T *d_decData;
        unsigned char *d_cmpBytes;
        cudaMalloc((void **)&d_decData, data_size * sizeof(T));
        cudaMalloc((void **)&d_cmpBytes, compressed_size);

        cudaMemcpy(d_cmpBytes, compressed_data, compressed_size,
                   cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // d_cmpBytes and d_decData are device pointers
        cuSZp_decompress(d_decData, d_cmpBytes, data_size, compressed_size, tol,
                         dataType, encodingMode, stream);

        decompressed_data = malloc(data_size * sizeof(T));
        cudaMemcpy(decompressed_data, d_decData, data_size * sizeof(T),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_decData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);

        return true;
    }
};

#endif
// GPU_SZ_COMPRESSOR_HPP
