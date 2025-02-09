rm -rf tmp
cd build
cmake ..
make
cd ..
mkdir tmp
cd tmp

vec_size=$1
precision=$2

echo
echo " ================ RUNNING ================"
echo

echo "Running CPU"
./../build/fft $vec_size
# dot -Kfdp -n -Tpng -o graph_ordered.png graph.dot 
# dot -Tpng -o graph.png graph.dot 

echo "Running python"
python3 ../fft.py $vec_size

echo
echo " ================ EVALUATION ================"
echo

echo "precision=$precision"

echo
echo "c++ CPU vs Python"
python3 ../compare.py fft_output_python.txt output_cpp_seq.txt $precision

echo
echo "cuda GPU vs Python"
python3 ../compare.py fft_output_python.txt output_cpp_gpu.txt $precision

echo
echo "c++ CPU vs cuda GPU"
python3 ../compare.py output_cpp_seq.txt output_cpp_gpu.txt $precision

echo
echo "c++ CPU vs cuda GPU shared"
python3 ../compare.py output_cpp_seq.txt output_cpp_gpu_shared.txt $precision