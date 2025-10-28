#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include "types.hpp"

float iou(const Box &b1, const Box &b2);
void nms(std::vector<Box> &bboxes, float conf_threshold = 0.25f, bool is_sorted = false);
