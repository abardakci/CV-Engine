#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "cv_types.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

// Hata kontrol makrosu
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

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

void drawTrack(cv::Mat& target, Track& track);

inline double euclidianDistance(int x1, int y1, int x2, int y2)
{
    double dx = std::pow(x1 - x2, 2);
    double dy = std::pow(y1 - y2, 2);
    double d = std::sqrt(dx + dy);

    return d;
};

float iou(Box& b1, Box& b2);
void nms(std::vector<Box>& bboxes, float conf_threshold = 0.25f, bool is_sorted = false);
