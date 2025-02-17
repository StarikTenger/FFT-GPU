#pragma once
#include "fl.h"
#include <sstream>

void img_to_freq(const fl *img_buff, fl *freq_buff, size_t width, size_t height);
void freq_to_img(const fl *img_buff, fl *freq_buff, size_t width, size_t height);

