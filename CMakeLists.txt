cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(FFT-GPU LANGUAGES CXX)

# set CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(gemm_cpu main_cpu.cpp)


# CUDA Libraries
# find_package(CUDA)
# if(CUDA_FOUND)
#   enable_language(CUDA)
#   include_directories(${CUDA_INCLUDE_DIRS} )
#   add_executable(gemm_cuda main_gpu.cu)
# endif()