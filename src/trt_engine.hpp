#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include "cuda_runtime.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

#include "cuda_utils.hpp"

#include "utils.hpp"

class Logger : public nvinfer1::ILogger 
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

};

class TrtEngine
{
public:
    TrtEngine(nvinfer1::ICudaEngine& engine);
    ~TrtEngine();
    int infer(const cv::Mat& input, cv::Mat& output);

private:
    std::unique_ptr<nvinfer1::IExecutionContext> m_ctx;
    
};
