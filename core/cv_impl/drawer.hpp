#pragma once

#include <opencv2/opencv.hpp>
#include "types.hpp"

void drawBox(cv::Mat &target, const Box &box);
void drawBoxes(cv::Mat &target, const std::vector<Box> &boxes);
// void drawTrack(cv::Mat &target, Track &track);
