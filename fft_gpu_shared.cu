#include "fft_gpu_shared.h"
#include "util.h"
#include "fl.h"

#include <cstdlib>
#include <iostream>
#include <complex>
#include <math.h>
#include <numbers>
#include <thrust/complex.h>
#include <chrono>

__global__ void fft_step_shared(const fl *buff_in, fl *buff_out, size_t N, size_t epoch, size_t log_bsize) {
    __shared__ fl shared_buff_1[2048];
    __shared__ fl shared_buff_2[2048];

    // Load from global memory
    int pos = (blockIdx.x>>(epoch * log_bsize)) * 1<<(log_bsize * (epoch + 1));
    int offset = blockIdx.x % (1<<(log_bsize * epoch));
    int global_idx = pos + offset + threadIdx.x * (1<<(log_bsize * epoch));
    shared_buff_1[threadIdx.x] = buff_in[global_idx];
    shared_buff_1[threadIdx.x + blockDim.x] = buff_in[global_idx + N];


    fl *ref1 = shared_buff_1;
    fl *ref2 = shared_buff_2;

    __syncthreads();
    for (size_t stride = 2; stride <= blockDim.x; stride <<= 1) {
        size_t i = threadIdx.x;
        size_t idx_1 = i;
        size_t idx_2 = (i ^ (stride / 2)) % blockDim.x;
        
        if (idx_1 > idx_2) {
            // Swap
            size_t temp = idx_1;
            idx_1 = idx_2;
            idx_2 = temp;
        }
        
        thrust::complex<fl> w = thrust::pow(
            M_E,
            thrust::complex<fl>(0, -2. * M_PI * global_idx / (stride * (1<<(log_bsize * epoch))))
        );

        thrust::complex<fl> res =
            thrust::complex<fl>(ref1[idx_1], ref1[idx_1 + blockDim.x]) +
            thrust::complex<fl>(ref1[idx_2], ref1[idx_2 + blockDim.x]) * w;


        ref2[i] = res.real();
        ref2[i + blockDim.x] = res.imag();
        swap(ref1, ref2);

        __syncthreads();
    }

    // Save back to global
    buff_out[global_idx] = ref1[threadIdx.x];
    buff_out[global_idx + N] = ref1[threadIdx.x + blockDim.x];
}

void fft_gpu_shared(const fl *buff_in, fl *buff_out, size_t N) {
    fl *buff1 = new fl[N * 2];
    fl *buff_to_delete = buff1;
    memcpy((void*)buff1, (void*)buff_in, (N * 2) * sizeof(fl));
    fl *buff2 = buff_out;
    

    // Reorder buffer
    size_t bitlen = log2(N);
    for (size_t i = 0; i < N; i++) {
        size_t j = 0;
        for (size_t k = 0; k < bitlen; k++) {
            j = (j << 1) | ((i >> k) & 1);
        }
        buff2[i] = buff1[j];
        buff2[N + i] = buff1[N + j];
    }

    swap(buff1, buff2);

    // Allocate buffers on GPU
    fl *buff_gpu1;
    fl *buff_gpu2;
    const size_t buff_size = N * 2 * sizeof(fl);

    cudaError_t err;

    err = cudaMalloc((void**)&buff_gpu1, buff_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for buff_gpu1: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&buff_gpu2, buff_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for buff_gpu2: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(buff_gpu1, buff1, buff_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying memory to buff_gpu1: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Define workspace topology
    size_t log_bsize = 5;
    size_t block_size = 1<<log_bsize;
	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(N / block_size, 1);



    for (size_t epoch = 0; epoch < bitlen / log_bsize; epoch++) {
        
        // cerr << "Epoch " << epoch << "; step=" << (1<<(log_bsize * epoch)) << "\n";

        fft_step_shared<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N, epoch, log_bsize);
        cudaDeviceSynchronize();

        swap(buff_gpu1, buff_gpu2);
    }

    cudaMemcpy(buff_out, buff_gpu1, buff_size, cudaMemcpyDeviceToHost);

    delete[] buff_to_delete;
}