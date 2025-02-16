#include "bitrev_reorder.h"
#include <math.h>

void bitrev_reorder(const fl *buff_in, fl *buff_out, size_t N) {
    size_t bitlen = log2(N);
    for (size_t i = 0; i < N; i++) {
        size_t j = 0;
        for (size_t k = 0; k < bitlen; k++) {
            j = (j << 1) | ((i >> k) & 1);
        }
        buff_out[i] = buff_in[j];
        buff_out[N + i] = buff_in[N + j];
    }
}