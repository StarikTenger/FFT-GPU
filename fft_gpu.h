#pragma once
#include "fl.h"
#include <sstream>

void fft_gpu(const fl *buff_in, fl *buff_out, size_t N);
void reverse_fft_gpu(const fl *buff_in, fl *buff_out, size_t N);