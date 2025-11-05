#pragma once

#include <iostream>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>

#include "types.hpp"
#include "timer.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

class IEngine; // forward declaration

/// Implements ESRGAN
class Upscaler
{
public:
    Upscaler(std::unique_ptr<IEngine> engine, const std::string& engine_path);
    ~Upscaler() = default;
    cv::Mat run(cv::Mat& input);

private:
    cv::Mat preproc(cv::Mat& mat);
    cv::Mat postproc(cv::Mat& mat);

private:
    std::unique_ptr<IEngine> engine_;

};
