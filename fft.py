import numpy as np

N = 8
array = np.zeros(N)
array[N//2:] = 1

print("Original array:")
print(array)

# Apply FFT
fft_result = np.fft.fft(array)

print("FFT result (real part):")
print(fft_result.real)

print("FFT result (imaginary part):")
print(fft_result.imag)

def serialize_fft_to_file(filename, fft_result, precision):
    with open(filename, 'w') as f:
        f.write(f"{len(fft_result)}\n")
        for real, imag in zip(fft_result.real, fft_result.imag):
            f.write(f"{real:.{precision}f}\t{imag:.{precision}f}\n")

# Serialize FFT result to file
serialize_fft_to_file('fft_output_python.txt', fft_result, 2)