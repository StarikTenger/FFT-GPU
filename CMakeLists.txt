cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(FFT-GPU LANGUAGES CXX CUDA)

# set CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})

# CUDA Libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(FFT_SOURCES fft_gpu.cu fft_gpu_shared.cu fft_cpu.cpp genbmp.cpp util.cpp bitrev_reorder.cpp)

# Main files
set(MAIN_FILES main.cu main_gpu.cu main_shared.cu)

# Add executables
foreach(MAIN_FILE ${MAIN_FILES})
    get_filename_component(EXECUTABLE_NAME ${MAIN_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${MAIN_FILE} ${FFT_SOURCES})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
endforeach()
