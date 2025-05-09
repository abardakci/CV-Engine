#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

#include <opencv2/opencv.hpp>

using timer = std::chrono::high_resolution_clock;

void getTiles(const cv::Mat& input, float* tiles, const int wTileNum, const int hTileNum, const int tileSize, const int overlap);

void printTime(const std::string &msg, timer::time_point start, timer::time_point end);

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output);
