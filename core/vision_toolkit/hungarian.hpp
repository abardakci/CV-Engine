#pragma once

#include <opencv2/opencv.hpp>
#include "types.hpp"

std::vector<int> hungarian(const std::vector<float>& cost_matrix, const int n);
