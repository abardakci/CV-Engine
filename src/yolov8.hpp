#pragma once

#include <iostream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "trt_engine.hpp"

class Yolov8
{
public:
    Yolov8(const std::string& path);
    ~Yolov8();

    cv::Mat preprocess(const cv::Mat& input, letterbox_t& letter);
    std::vector<Box> infer(const cv::Mat& input);
    std::vector<Box> postprocess(const cv::Mat& yolo_output, letterbox_t& letter, int h, int w);
    
private:
    TrtEngine trt_engine_;
    const int   kClassNum      = 80;
    const float kConfThreshold = 0.25;
    
};
