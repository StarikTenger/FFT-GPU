#pragma once

#include <iostream>
#include <complex>
#include <numbers>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "genbmp.h"

#include "fl.h"

using namespace std;

void print_buff(const fl *buff, size_t N);
void serialize_output(const fl *buff, size_t N, const string &filename, int precision = 2);