#pragma once

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "yolo_cfg.hpp"
#include "types.hpp"
#include "timer.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

class IEngine; // forward declaration

class Yolov8
{
public:
    Yolov8(const YoloConfig &cfg, std::unique_ptr<IEngine> engine);
    ~Yolov8() = default;

    cv::Mat preprocess(const cv::Mat &input, forge::letterbox_t &letter);
    std::vector<forge::Box> infer(const cv::Mat &input);
    std::vector<forge::Box> postprocess(const cv::Mat &yolo_output, forge::letterbox_t &letter, int h, int w);

private:
    YoloConfig cfg_;
    std::unique_ptr<IEngine> engine_;
    size_t input_size_;
    size_t output_size_;
    size_t preds_per_cell_;
    size_t output_cell_count_;
    std::vector<unsigned char> preproc_buffer_;
};
