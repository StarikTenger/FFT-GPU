#include "fft_gpu.h"
#include "util.h"
#include "fl.h"
#include "bitrev_reorder.h"

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <numbers>
#include <thrust/complex.h>
#include <chrono>

using namespace std;


/* Questions:
- In which memory are we working?
- Access is uncoalesced, how to fix it?
*/



__global__ void fft_step(const fl *buff_in, fl *buff_out, size_t N, size_t stride) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_1 = i;
    size_t idx_2 = (i ^ (stride / 2)) % N;
    
    if ((i % stride) >= stride / 2) {
        // Swap
        size_t temp = idx_1;
        idx_1 = idx_2;
        idx_2 = temp;
    }
    
    thrust::complex<fl> w = thrust::pow(
        M_E,
        thrust::complex<fl>(0, -2. * M_PI * i / stride)
    );

    thrust::complex<fl> res =
        thrust::complex<fl>(buff_in[idx_1], buff_in[idx_1 + N]) +
        thrust::complex<fl>(buff_in[idx_2], buff_in[idx_2 + N]) * w;



    buff_out[i] = res.real();
    buff_out[i + N] = res.imag();
}

__global__ void conjugate(const fl *buff_in, fl *buff_out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    buff_out[i] = buff_in[i];
    buff_out[i + N] = -buff_in[i + N];
}

__global__ void divide_N(const fl *buff_in, fl *buff_out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    buff_out[i] = buff_in[i] / N;
    buff_out[i + N] = buff_in[i + N] / N;
}

void fft_gpu(const fl *buff_in, fl *buff_out, size_t N) {
    fl *buff1 = new fl[N * 2];
    fl *buff_to_delete = buff1;
    memcpy((void*)buff1, (void*)buff_in, (N * 2) * sizeof(fl));
    fl *buff2 = buff_out;
    

    // Reorder buffer
    bitrev_reorder(buff1, buff2, N);
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
    size_t block_size = 32;
	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(N / block_size, 1);

    for (size_t stride = 2; stride <= N; stride <<= 1) {

        fft_step<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N, stride);
        cudaDeviceSynchronize();

        swap(buff_gpu1, buff_gpu2);
    }

    cudaMemcpy(buff_out, buff_gpu1, buff_size, cudaMemcpyDeviceToHost);

    delete[] buff_to_delete;
}



void reverse_fft_gpu(const fl *buff_in, fl *buff_out, size_t N) {
    fl *buff1 = new fl[N * 2];
    fl *buff_to_delete = buff1;
    memcpy((void*)buff1, (void*)buff_in, (N * 2) * sizeof(fl));
    fl *buff2 = buff_out;
    

    // Reorder buffer
    bitrev_reorder(buff1, buff2, N);
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
    size_t block_size = BLOCK_SIZE;
	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(N / block_size, 1);

    conjugate<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N);
    cudaDeviceSynchronize();
    swap(buff_gpu1, buff_gpu2);

    for (size_t stride = 2; stride <= N; stride <<= 1) {

        fft_step<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N, stride);
        cudaDeviceSynchronize();

        swap(buff_gpu1, buff_gpu2);
    }

    conjugate<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N);
    cudaDeviceSynchronize();
    swap(buff_gpu1, buff_gpu2);

    divide_N<<<dimGrid, dimBlock>>>(buff_gpu1, buff_gpu2, N);
    cudaDeviceSynchronize();
    swap(buff_gpu1, buff_gpu2);

    cudaMemcpy(buff_out, buff_gpu1, buff_size, cudaMemcpyDeviceToHost);

    delete[] buff_to_delete;
}