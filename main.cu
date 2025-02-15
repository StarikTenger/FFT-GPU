#include "fl.h"
#include "genbmp.h"
#include "util.h"
#include "fft_cpu.h"
#include "fft_gpu.h"
#include "fft_gpu_shared.h"

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

    // CPU
    {
        cout << "\nRunning fft cpu" << endl;
        auto start = chrono::high_resolution_clock::now();

        fft_cpu(buff_in, buff_out, N, graph_stream);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Time taken for cpu: " << elapsed.count() * 1000 << " ms" << endl;

        // Serialize output
        serialize_output(buff_out, N, "output_cpp_seq.txt", 6);
    }
   
    // =========================================================================

    graph_stream << "}";

    // Dump graph to file
    ofstream graph_file("graph.dot");
    graph_file << graph_stream.str();
    graph_file.close();

    // {
    //     // Shared topology test
    //     int log_bsize = 1;
    //     int bsize = 1<<log_bsize;
    //     int N = 16;

    //     for (int epoch = 0; epoch < 4; epoch++) {
    //         cout << "Epoch " << epoch << "\n";

    //         vector<set<pair<int, int>>> mapping(N);

    //         for (int b = 0; b < N / bsize; b++) {
    //             int pos = (b>>(epoch * log_bsize)) * 1<<(log_bsize * (epoch + 1));
    //             int offset = b % (1<<(log_bsize * epoch));
    //             cout << "b = " << b << "; pos = " << pos << "; offset = " << offset << endl;
    //             for (int i = 0; i < bsize; i++) {
    //                 mapping[pos + offset + i * (1<<(log_bsize * epoch))].insert({b, i});
    //             }
    //         }

    //         for (int i = 0; i < N; i++) {
    //             cout << i << "\t: ";
    //             for (auto it = mapping[i].begin(); it != mapping[i].end(); ++it) {
    //                 if (it != mapping[i].begin()) {
    //                     cout << ", ";
    //                 }
    //                 cout << it->first << " : " << it->second;
    //             }
    //             cout << endl;
    //         }
    //         cout << endl;
    //     }      

    // }
}