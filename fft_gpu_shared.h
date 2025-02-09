#pragma once
#include "fl.h"
#include <sstream>

void fft_gpu_shared(const fl *buff_in, fl *buff_out, size_t N);