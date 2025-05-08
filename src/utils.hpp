#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

using timer = std::chrono::high_resolution_clock;

void getTiles(const cv::Mat& input, float* tiles, const int tileSize = 128);

void printTime(const std::string &msg, timer::time_point start, timer::time_point end);

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output);
