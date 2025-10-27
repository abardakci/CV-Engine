#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include "types.hpp"

float iou(Box &b1, Box &b2);
void nms(std::vector<Box> &bboxes, float conf_threshold = 0.25f, bool is_sorted = false);
