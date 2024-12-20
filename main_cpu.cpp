#include "fl.h"
#include "genbmp.h"
#include "util.h"

#include <cstdlib>
#include <iostream>
#include <complex>
#include <numbers>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace std;

/* v1 * v2 -> v3 */
void complex_mul(const fl v1[2], const fl v2[2], fl v3[2]) {
    v3[0] = v1[0] * v2[0] - v1[1] * v2[1];
    v3[1] = v1[0] * v2[1] + v1[1] * v2[0];
}


/*  
 * buffers are of size 2N, first half - real part, second - complex
 * stride is 2^i, where i is step number
 */
void fft_step(const fl *buff_in, fl *buff_out, size_t N, size_t stride, stringstream &graph_stream) {

    size_t step = 0;
    for (size_t i = stride; i > 1; i >>= 1) {
        step++;
    }

    for (size_t i = 0; i < N; i++) {
        size_t idx_1 = i;
        size_t idx_2 = (i ^ (stride / 2)) % N;
        if ((i % stride) >= stride / 2) {
            swap(idx_1, idx_2);
        }
        
        // graph_stream 
        //      << "" << step -1   << "." << idx_1 << " -> "
        //      << "" << step << "." << i
        //      << "\n";
        // graph_stream 
        //      << "" << step - 1 << "." << idx_2 << " -> "
        //      << "" << step << "." << i
        //      << "[color=red,label=\"W" << stride << " " << i << "\"]\n";

        complex<fl> w = std::pow(
            numbers::e_v<fl>,
            complex<fl>(0, -2. * numbers::pi_v<fl> * i / stride)
        );

        complex<fl> res =
            complex<fl>(buff_in[idx_1], buff_in[idx_1 + N]) +
            complex<fl>(buff_in[idx_2], buff_in[idx_2 + N]) * w;

        buff_out[i] = res.real();
        buff_out[i + N] = res.imag();
    }
}

void fft(const fl *buff_in, fl *buff_out, size_t N, stringstream &graph_stream) {

    fl *buff1 = new fl[N * 2];
    fl *buff_to_delete = buff1;
    memcpy((void*)buff1, (void*)buff_in, (N * 2) * sizeof(fl));
    fl *buff2 = buff_out;
    bool need_swap = 1;

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
    need_swap = !need_swap;


    for (size_t stride = 2; stride <= N; stride <<= 1) {
        //cout << "stride = " << stride << endl;
        fft_step(buff1, buff2, N, stride, graph_stream);

        //cout << endl;

        swap(buff1, buff2);
        need_swap = !need_swap;
    }

    if (need_swap) {
        memcpy(buff_out, buff1, (N * 2) * sizeof(fl));
    }
    delete buff_to_delete;
}

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

    // Print buff_in
    // print_buff(buff_in, N);

    // FFT
    auto start = chrono::high_resolution_clock::now();

    fft(buff_in, buff_out, N, graph_stream);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Time taken for cpu: " << elapsed.count() * 1000 << " ms" << endl;

    // Print buff_out
    // print_buff(buff_out, N);

    graph_stream << "}";

    // Dump graph to file
    ofstream graph_file("graph.dot");
    graph_file << graph_stream.str();
    graph_file.close();

    // Serialize output
    serialize_output(buff_out, N, "output_cpp_seq.txt", 6);
}