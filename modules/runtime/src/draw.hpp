#pragma once

#include <opencv2/opencv.hpp>
#include "utils/types.hpp"

void drawBox(cv::Mat &target, const Box &box, const std::string& text);
