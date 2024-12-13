rm -rf tmp
cd build
cmake ..
make
cd ..
mkdir tmp
cd tmp

./../build/gemm_cpu
dot -Kfdp -n -Tpng -o graph_ordered.png graph.dot 
dot -Tpng -o graph.png graph.dot 

python3 ../fft.py
python3 ../compare.py fft_output_python.txt output_cpp_seq.txt