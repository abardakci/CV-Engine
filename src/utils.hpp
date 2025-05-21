#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "cv_types.hpp"

using timer = std::chrono::high_resolution_clock;

void printTime(const std::string &msg, timer::time_point start, timer::time_point end);

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output);

cv::Mat letterbox(cv::Mat input, letterbox_t& letter, int w, int h);

inline int clamp(int num, int min, int max)
{
    return num < min ? min : (num > max ? max : num);
}

void drawBox(cv::Mat& target, const Box& box);

void drawBoxes(cv::Mat& target, const std::vector<Box>& boxes);

inline double euclidianDistance(int x1, int y1, int x2, int y2)
{
    double dx = std::pow(x1 - x2, 2);
    double dy = std::pow(y1 - y2, 2);
    double d = std::sqrt(dx + dy);

    return d;
};
