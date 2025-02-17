#pragma once
#include "fl.h"
#include <sstream>

#define BLOCK_SIZE 1024

void fft_gpu(const fl *buff_in, fl *buff_out, size_t N);
void reverse_fft_gpu(const fl *buff_in, fl *buff_out, size_t N);