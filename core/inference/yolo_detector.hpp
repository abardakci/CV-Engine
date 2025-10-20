#pragma once

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "engine_interface.hpp"
#include "types.hpp"
#include "postprocess.hpp"

class Yolov8
{
public:
    Yolov8(const std::string &path, std::unique_ptr<IEngine> engine);
    ~Yolov8();

    cv::Mat preprocess(const cv::Mat &input, letterbox_t &letter);
    std::vector<Box> infer(const cv::Mat &input);
    std::vector<Box> postprocess(const cv::Mat &yolo_output, letterbox_t &letter, int h, int w);

private:
    std::unique_ptr<IEngine> engine_;
    size_t input_size_;
    size_t output_size_;
    const int kClassNum = 80;
    const float kNmsThreshold = 0.45;
    const float kConfThreshold = 0.25;

};
