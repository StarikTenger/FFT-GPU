#include "image_freq.h"
#include "fft_gpu.h"

#include <cstdlib>
#include <iostream>
#include <thrust/complex.h>

using namespace std;

__global__ void untangle(const fl *buff_in, fl *buff_out, size_t width, size_t height) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y;
    size_t offset = row * blockDim.x * gridDim.x;

    fl h1_r = buff_in[offset + i];
    fl h1_i = buff_in[offset + i + width];
    fl h2_r = buff_in[offset + width - 1 - i];
    fl h2_i = buff_in[offset + width - 1 - i + width];
    
    fl f_r = (h1_r + h2_r) / 2;
    fl f_i = (h1_i - h2_i) / 2;
    fl g_r = (h1_i + h2_i) / 2;
    fl g_i = (h2_r - h1_r) / 2;

    buff_out[offset + i] = f_r;
    buff_out[offset + i + width] = g_r;
    buff_out[offset + width - 1 - i] = -f_i;
    buff_out[offset + width - 1 - i + width] = -g_i;
    
}

void serial_untangle(const fl *buff_in, fl *buff_out, size_t width, size_t height) {
    for (size_t i = 0; i < height; i += 2) {
        size_t offset = i * width;
        for (size_t j = 0; j < width / 2; j++) {
            fl h1_r = buff_in[offset + j];
            fl h1_i = buff_in[offset + j + width];
            fl h2_r = buff_in[offset + width - 1 - j];
            fl h2_i = buff_in[offset + width - 1 - j + width];
            
            fl f_r = (h1_r + h2_r) / 2;
            fl f_i = (h1_i - h2_i) / 2;
            fl g_r = (h1_i + h2_i) / 2;
            fl g_i = (h2_r - h1_r) / 2;

            buff_out[offset + j] = f_r;
            buff_out[offset + j + width] = g_r;
            buff_out[offset + width - 1 - j] = -f_i;
            buff_out[offset + width - 1 - j + width] = -g_i;
        }
    }
}

// void img_to_freq_packed(const fl *img_buff, fl *freq_buff, size_t width, size_t height) {
//     // First, doing sequentialy, do better later

//     // Calculate FFT for each tangled pair
//     fl *buff1 = new float[width * height];
//     for (int i = 0; i < height; i+=2) {
//         size_t offset = i * width * 2;
//         fft_gpu(img_buff + offset, buff1 + offset, width);
//     }

//     fl *buff2 = new float[width * height];

//     // Untangle
//     serial_untangle(buff1, buff2, width, height);

// }


// No packing version
void img_to_freq(const fl *img_buff, fl *freq_buff, size_t width, size_t height) {

    // Copy image to buffer

    cout << "copy:" << endl;

    fl *buff1 = new fl[width * height * 2];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            buff1[i * width * 2 + j] = img_buff[i * width + j];
            buff1[i * width * 2 + j + width] = 0;
        }
    }

    cout << "fft rows:" << endl;

    // FFT on rows
    fl *buff2 = new fl[width * height * 2];

    for (int i = 0; i < height; i++) {
        size_t offset = i * width * 2;
        fft_gpu(buff1 + offset, buff2 + offset, width);
    }
    swap(buff1, buff2);

    cout << "reorede" << endl;

    // Reorder

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            buff2[j * height * 2 + i] = buff1[i * width * 2 + j];
            buff2[j * height * 2 + i + height] = buff1[i * width * 2 + j + width];
        }
    }
    swap(buff1, buff2);


    cout << "fft columns:" << endl;

    // FFT on columns

    for (int i = 0; i < width; i++) {
        size_t offset = i * height * 2;
        fft_gpu(buff1 + offset, buff2 + offset, height);
    }

    cout << "copy back" << endl;

    // Copy back

    memcpy((void*)freq_buff, (void*)buff2, width * height * 2 * sizeof(fl));

    delete[] buff1;
    delete[] buff2;
}

void freq_to_img(const fl *freq_buff, fl *img_buff, size_t width, size_t height) {
    // Reverse fft on columns
    fl *buff1 = new fl[width * height * 2];

    cout << "1" << endl;

    for (int i = 0; i < width; i++) {
        size_t offset = i * height * 2;
        reverse_fft_gpu(freq_buff + offset, buff1 + offset, height);
        //cout << "i: " << i << endl;
    }

    cout << "2" << endl;

    // Reorder
    fl *buff2 = new fl[width * height * 2];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            buff2[i * width * 2 + j] =  buff1[j * height * 2 + i];
            buff2[i * width * 2 + j + width] = buff1[j * height * 2 + i + height];
        }
    }
    swap(buff1, buff2);

    cout << "3" << endl;

    // Reverse fft on rows
    for (int i = 0; i < height; i++) {
        size_t offset = i * width * 2;
        reverse_fft_gpu(buff1 + offset, buff2 + offset, width);
    }

    cout << "4" << endl;

    // Copy back
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img_buff[i * width + j] = buff2[i * width * 2 + j];
        }
    }

    delete[] buff1;
    delete[] buff2;
}