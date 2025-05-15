#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>

#include <opencv2/opencv.hpp>

using timer = std::chrono::high_resolution_clock;

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

void printTime(const std::string &msg, timer::time_point start, timer::time_point end);

int loadBinaryFromFile(const std::string& fileName, std::vector<char>& output);

cv::Mat letterbox(cv::Mat input, letterbox_t& letter, int w, int h);

inline int clamp(int num, int min, int max)
{
    return num < min ? min : (num > max ? max : num);
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
    
    void print()
    {
        std::cout << "x1: " << x1;
        std::cout << ", y1: " << y1;
        std::cout << ", x2: " << x2;
        std::cout << ", y2: " << y2 << "\n";
        std::cout << "class id: " << class_id << "\n";
        std::cout << "conf score: " << conf_score << "\n";
    }
};

void draw_box(cv::Mat& target, const Box& box);

void draw_boxes(cv::Mat& target, const std::vector<Box>& boxes);