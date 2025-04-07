#ifndef GPU_ZFP_COMPRESSOR_HPP
#define GPU_ZFP_COMPRESSOR_HPP

#include "GeneralCompressor.hpp"
#include "zfp.h"
#include <cuda_runtime.h>
#include <iostream>

// Here is a little problem:
// ZFP only support fix rate mode when using CUDA
// So we have to manually do conversion

template <typename T>
class GPUZFPCompressor : public GeneralCompressor<T>
{
public:
    bool compress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                  T *original_data, void *&compressed_data,
                  size_t &compressed_size) override
    {
        size_t original_size = 1;
        for (mgard_x::DIM i = 0; i < D; i++)
            original_size *= shape[i];

        zfp_type type;
        if (std::is_same<T, double>::value)
            type = zfp_type_double;
        else if (std::is_same<T, float>::value)
            type = zfp_type_float;
        else
        {
            std::cout << "wrong dtype\n";
            exit(-1);
        }

        T *d_original_data = nullptr;
        cudaMalloc(&d_original_data, sizeof(T) * original_size);
        cudaMemcpy(d_original_data, original_data, sizeof(T) * original_size,
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        zfp_field *field = nullptr;
        if (D == 1)
            field = zfp_field_1d(d_original_data, type, shape[0]);
        else if (D == 2)
            field = zfp_field_2d(d_original_data, type, shape[1], shape[0]);
        else if (D == 3)
            field = zfp_field_3d(d_original_data, type, shape[2], shape[1], shape[0]);
        else if (D == 4)
            field = zfp_field_4d(d_original_data, type, shape[3], shape[2], shape[1],
                                 shape[0]);
        else
        {
            std::cout << "wrong D\n";
            exit(-1);
        }

        zfp_stream *zfp = zfp_stream_open(NULL);
        // COnverse tol to rate
        double max_range = 4780302.524170;
        double rate = std::log2(max_range / tol);
        zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false);

        size_t bufsize = zfp_stream_maximum_size(zfp, field);
        void *d_compressed_data = nullptr;
        cudaMalloc(&d_compressed_data, bufsize);
        cudaDeviceSynchronize();

        std::cout << "bufsize = " << bufsize << std::endl;
        if (!d_compressed_data)
        {
            std::cerr << "Failed to allocate device memory for compression." << std::endl;
            exit(-1);
        }

        bitstream *stream = stream_open(d_compressed_data, bufsize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        if (!zfp_stream_set_execution(zfp, zfp_exec_cuda))
        {
            std::cout << "zfp-cuda not available\n";
            exit(-1);
        }

        compressed_size = zfp_compress(zfp, field);
        cudaDeviceSynchronize();

        if (compressed_size == 0)
        {
            std::cout << "zfp-cuda compress error\n";
            exit(-1);
        }

        void *host_buffer = malloc(compressed_size);
        if (!host_buffer)
        {
            std::cerr << "Failed to allocate host memory for compression." << std::endl;
            exit(-1);
        }
        cudaMemcpy(host_buffer, d_compressed_data, compressed_size,
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        compressed_data = host_buffer;

        cudaFree(d_original_data);
        cudaFree(d_compressed_data);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        return true;
    }

    bool decompress(mgard_x::DIM D, std::vector<mgard_x::SIZE> shape, double tol,
                    void *compressed_data, size_t compressed_size,
                    void *&decompressed_data) override
    {
        size_t original_size = 1;
        for (mgard_x::DIM i = 0; i < D; i++)
            original_size *= shape[i];

        zfp_type type;
        if (std::is_same<T, double>::value)
            type = zfp_type_double;
        else if (std::is_same<T, float>::value)
            type = zfp_type_float;
        else
        {
            std::cout << "wrong dtype\n";
            exit(-1);
        }

        T *d_decompressed_data = nullptr;
        cudaMalloc(&d_decompressed_data, sizeof(T) * original_size);
        uint8_t *d_compressed_data = nullptr;
        cudaMalloc(&d_compressed_data, compressed_size);
        cudaMemcpy(d_compressed_data, compressed_data, compressed_size,
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        zfp_field *field = nullptr;
        if (D == 1)
            field = zfp_field_1d(d_decompressed_data, type, shape[0]);
        else if (D == 2)
            field = zfp_field_2d(d_decompressed_data, type, shape[1], shape[0]);
        else if (D == 3)
            field = zfp_field_3d(d_decompressed_data, type, shape[2], shape[1], shape[0]);
        else if (D == 4)
            field = zfp_field_4d(d_decompressed_data, type, shape[3], shape[2], shape[1],
                                 shape[0]);
        else
        {
            std::cout << "wrong D\n";
            exit(-1);
        }

        zfp_stream *zfp = zfp_stream_open(NULL);
        double max_range = 4780302.524170;
        double rate = std::log2(max_range / tol);
        zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false);

        bitstream *stream = stream_open(d_compressed_data, compressed_size);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        if (!zfp_stream_set_execution(zfp, zfp_exec_cuda))
        {
            std::cout << "zfp-cuda not available\n";
            exit(-1);
        }

        int status = zfp_decompress(zfp, field);
        cudaDeviceSynchronize();

        if (!status)
        {
            std::cout << "zfp-cuda decompress error\n";
            exit(-1);
        }

        decompressed_data = new T[original_size];
        cudaMemcpy(decompressed_data, d_decompressed_data, sizeof(T) * original_size,
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaFree(d_decompressed_data);
        cudaFree(d_compressed_data);
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        return true;
    }
};

#endif
