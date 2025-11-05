#pragma once

#include <opencv2/opencv.hpp>
#include "types.hpp"

void drawBox(cv::Mat &target, const forge::Box &box, const std::string& text);
