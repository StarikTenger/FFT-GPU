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
#include <thrust/complex.h>

using namespace std;

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
}

void fft_cpu(const fl *buff_in, fl *buff_out, size_t N, stringstream &graph_stream) {

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


    for (size_t stride = 2; stride <= N; stride <<= 1) {
        //cout << "stride = " << stride << endl;
        fft_step(buff1, buff2, N, stride, graph_stream);

        //cout << endl;

        swap(buff1, buff2);
    }

    memcpy(buff_out, buff1, (N * 2) * sizeof(fl));

    delete buff_to_delete;
}