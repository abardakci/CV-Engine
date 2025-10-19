#pragma once

#include <opencv2/opencv.hpp>
#include "types.hpp"

void drawBox(cv::Mat &target, const Box &box, const std::string &text = "empty text");
