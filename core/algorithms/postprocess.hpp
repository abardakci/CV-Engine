#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "types.hpp"

cv::Mat letterbox(cv::Mat input, letterbox_t &letter, int w, int h);

inline int clamp(int num, int min, int max)
{
    return num < min ? min : (num > max ? max : num);
}

inline double euclidianDistance(int x1, int y1, int x2, int y2)
{
    double dx = std::pow(x1 - x2, 2);
    double dy = std::pow(y1 - y2, 2);
    double d = std::sqrt(dx + dy);

    return d;
};

float iou(Box &b1, Box &b2);
void nms(std::vector<Box> &bboxes, float conf_threshold = 0.25f, bool is_sorted = false);
