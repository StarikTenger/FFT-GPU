FFT implementation on CUDA

# Repository Organization

## Source code

Algorithms themselves are found in files:
- fft_cpu.cpp
- fft_gpu.cu
- fft_gpu_shared.cu

In 3 main files performance evaluation for each algorithm happens:
- main.cu
- main_gpu.cu
- main_shared.cu

main_img.cpp is responsible for image processing

image_freq.cu contains image processing pipeline

other files are of less importance

## Scripts

`run.sh <size> <epsilon>` measures elapsed time for all algorithms and evaluates their correction

`performance_compare.sh` measures average elapsed time for different algorithms and various vector sizes

`performance_global.sh` measures average elapsed time for different block sizes of shared memory (need to uncomment line 82 and comment line 81 in *main_gpu.cu*)

## Experiments

Experiments results plots of which can be found in the report are given in 2 csv files