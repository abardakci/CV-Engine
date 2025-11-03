#pragma once

#include "types.hpp"
#include <vector>

std::vector<int> hungarian(const float* cost_data, const int rows, const int cols);
