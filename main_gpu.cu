#include "util.h"
#include "fl.h"
#include "fft_gpu.h"

#include <cstdlib>
#include <iostream>
#include <complex>
#include <math.h>
#include <numbers>
#include <thrust/complex.h>
#include <chrono>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    size_t N = std::atoi(argv[1]);
    if (N & (N - 1)) {
        std::cerr << "N must be a power of 2" << std::endl;
        return 1;
    }

    fl *buff_in = new fl[N * 2];
    fl *buff_out = new fl[N * 2];

    for (size_t i = 0; i < N; i++) {
        buff_in[i] = i < N / 2 ? 0 : 1;
        buff_in[i + N] = 0;
    }

    // Print buff_in
    // print_buff(buff_in, N);

    ///// FFT /////

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto chrono_start = chrono::high_resolution_clock::now();


    cudaEventRecord(start);

    fft_gpu(buff_in, buff_out, N);

    auto chrono_end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = chrono_end - chrono_start;
    cout << "FFT execution time (chrono): " << elapsed.count() * 1000 << " ms" << endl;

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time taken for gpu: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ///////////////

    // Print buff_out
    // print_buff(buff_out, N);

    // Serialize output
    serialize_output(buff_out, N, "output_cpp_gpu.txt", 6);

    delete[] buff_in;
    delete[] buff_out;

    return 0;
}