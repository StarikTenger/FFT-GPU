#!/bin/bash

# Define the vector sizes (powers of 2)
# vector_sizes=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
vector_sizes=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# Define the programs to run
programs=("main_gpu")

# Output file
output_file="performance_global.csv"

# Write the header to the CSV file
#echo "algorithm,vector_size,elapsed_time" > $output_file

# Loop over each vector size
for size in "${vector_sizes[@]}"; do
    # Loop over each program
    for program in "${programs[@]}"; do
        # Run each program 5 times
        for run in {1..5}; do
            ./build/$program $size >> $output_file
        done
    done
done
