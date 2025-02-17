#include "image_freq.h"
#include "genbmp.h"

#include <iostream>

using namespace std;

int main() {
    // Open image from a file
    const char* filename = "input.bmp";

    int width, height;
    unsigned char *image = readBMP(filename, width, height);

    cout << "bmp read" << endl;
    cout << "width: " << width << " height: " << height << endl;

    // Extract the red channel
    fl *red_buff = new fl[width * height];
    fl *green_buff = new fl[width * height];
    fl *blue_buff = new fl[width * height];
    for (int i = 0; i < width * height; i++) {
        red_buff[i] = image[i * BYTES_PER_PIXEL];
        green_buff[i] = image[i * BYTES_PER_PIXEL + 1];
        blue_buff[i] = image[i * BYTES_PER_PIXEL + 2];
    }

    cout << "red channel extracted" << endl;

    // Perform FFT
    fl *red_freq_buff = new fl[width * height * 2];
    img_to_freq(red_buff, red_freq_buff, width, height);

    fl *green_freq_buff = new fl[width * height * 2];
    img_to_freq(green_buff, green_freq_buff, width, height);

    fl *blue_freq_buff = new fl[width * height * 2];
    img_to_freq(blue_buff, blue_freq_buff, width, height);


    cout << "fft completed" << endl;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = j * height * 2 + i;
            if (i > height / 2 || j > width / 2) {
                red_freq_buff[idx] = 0;
                red_freq_buff[idx + height] = 0;

                green_freq_buff[idx] = 0;
                green_freq_buff[idx + height] = 0;

                blue_freq_buff[idx] = 0;
                blue_freq_buff[idx + height] = 0;
            }
        }
    }

    // Perform IFFT
    fl *img_buff_out_red = new fl[width * height];
    freq_to_img(red_freq_buff, img_buff_out_red, width, height);

    fl *img_buff_out_green = new fl[width * height];
    freq_to_img(green_freq_buff, img_buff_out_green, width, height);

    fl *img_buff_out_blue = new fl[width * height];
    freq_to_img(blue_freq_buff, img_buff_out_blue, width, height);

    cout << "ifft completed" << endl;

    // Save the image
    unsigned char *image_out = new unsigned char[width * height * BYTES_PER_PIXEL];
    for (int i = 0; i < width * height; i++) {
        image_out[i * BYTES_PER_PIXEL] = img_buff_out_red[i];
        image_out[i * BYTES_PER_PIXEL + 1] = img_buff_out_green[i];
        image_out[i * BYTES_PER_PIXEL + 2] = img_buff_out_blue[i];
    }

    cout << "img converted" << endl;

    generateBitmapImage(image_out, height, width, "output.bmp");

    cout << "img saved" << endl;
}