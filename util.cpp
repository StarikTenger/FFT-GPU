#include "util.h"

using namespace std;

void print_buff(const fl *buff, size_t N) {
    std::cerr << std::fixed << std::showpoint;
    std::cerr << std::setprecision(2);
    for (size_t i = 0; i < N; i++) {
        cerr << buff[i] << "\t";
    }
    cerr << endl;
    for (size_t i = 0; i < N; i++) {
        cerr << buff[i + N] << "\t";
    }
    cerr << endl;
}

void serialize_output(const fl *buff, size_t N, const string &filename, int precision) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    file << std::fixed << std::showpoint;
    file << std::setprecision(precision);

    file << N << endl;

    for (size_t i = 0; i < N; i++) {
        file << buff[i] << "\t";
        file << buff[i + N] << "\n";
    }
    file << endl;

    file.close();
}