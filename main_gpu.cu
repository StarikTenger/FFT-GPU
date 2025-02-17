#include "fl.h"
#include "genbmp.h"
#include "util.h"
#include "fft_gpu.h"

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <set>
#include <vector>

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

    for (int i = 0; i < N; i++) {
        buff_in[i] = i < N / 2 ? 0 : 1;
        buff_in[i + N] = 0;
    }

    size_t steps = 0;
    for (size_t stride = 2; stride <= N; stride <<= 1) {
        steps++;
    }

    // Initialize graph

    stringstream graph_stream;
    graph_stream << "digraph G {\n";

    for (size_t step = 0; step <= steps; step++) {
        for (size_t i = 0; i < N; i++) {
            graph_stream << step << "." << i << " [pos=\"" << step * 3 << "," << (N - i) << "!\"];\n";
        }
    }

    // ================================== FFT ==================================


    // GPU
    {
        cerr << "\nRunning fft gpu" << endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        auto chrono_start = chrono::high_resolution_clock::now();


        cudaEventRecord(start);

        fft_gpu(buff_in, buff_out, N);

        auto chrono_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = chrono_end - chrono_start;
        cerr << "FFT execution time (chrono): " << elapsed.count() * 1000 << " ms" << endl;

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cerr << "Time taken for gpu: " << milliseconds << " ms" << std::endl;
        //std::cout << "GPU,\t\t" << N << ",\t" << milliseconds << std::endl;
        std::cout << BLOCK_SIZE << ",\t\t" << N << ",\t" << milliseconds << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Serialize output
        serialize_output(buff_out, N, "output_cpp_gpu.txt", 6);

        swap(buff_in, buff_out);
        reverse_fft_gpu(buff_in, buff_out, N);
        serialize_output(buff_out, N, "output_cpp_gpu_reversed.txt", 6);

    }

   
    // =========================================================================
}