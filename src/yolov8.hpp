#pragma once

#include <iostream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "trt_engine.hpp"

class Yolov8
{
public:
    Yolov8();
    ~Yolov8();

    cv::Mat pre_process(const cv::Mat& input);
    std::vector<Box> infer(const cv::Mat& input);
    std::vector<Box> post_process(const cv::Mat& yolo_output, int h, int w);

private:
    TrtEngine m_trt_engine;

    int   num_of_class   = 80;
    float conf_threshold = 0.45;
    
};
