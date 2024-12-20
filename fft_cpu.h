#pragma once
#include "fl.h"
#include <sstream>

void fft_cpu(const fl *buff_in, fl *buff_out, size_t N, std::stringstream &graph_stream);