#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>

#include <opencv2/opencv.hpp>

using timer = std::chrono::high_resolution_clock;

void printTime(const std::string &msg, timer::time_point start, timer::time_point end);

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output);

cv::Mat letterbox(cv::Mat input, int w, int h);

inline int clamp(int num, int min, int max)
{
    return min ? num < min : (max ? num > max : num);
} 

class Box
{
public:
    Box(int x1, int y1, int x2, int y2, int class_id, float conf_score)
    {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->class_id = class_id;
        this->conf_score = conf_score;
    }

    int x1;
    int y1;
    int x2;
    int y2;
    int class_id;
    float conf_score;

};

void draw_box(cv::Mat& target, const Box& box);

void draw_boxes(cv::Mat& target, const std::vector<Box>& boxes);