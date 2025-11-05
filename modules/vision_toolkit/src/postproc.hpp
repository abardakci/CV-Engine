#pragma once

#include "types.hpp"
#include <vector>
#include <memory>
#include <algorithm>

float iou(const forge::Box &b1, const forge::Box &b2);
void nms(std::vector<forge::Box> &bboxes, float conf_threshold = 0.25f, bool is_sorted = false);
