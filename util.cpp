#include "util.h"

using namespace std;

void print_buff(const fl *buff, size_t N) {
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(2);
    for (size_t i = 0; i < N; i++) {
        cout << buff[i] << "\t";
    }
    cout << endl;
    for (size_t i = 0; i < N; i++) {
        cout << buff[i + N] << "\t";
    }
    cout << endl;
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