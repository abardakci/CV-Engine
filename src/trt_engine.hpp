#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include "cuda_runtime.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

#include "utils.hpp"

class Yolov8; // forward declaration

class Logger : public nvinfer1::ILogger 
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

};

class TrtEngine
{
public:
    TrtEngine(const std::string& engine_path);
    ~TrtEngine();
    int infer(float* input, float* output);

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* ctx_;

    std::string input_name_;
    std::string output_name_;

    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;

    size_t input_size_; 
    size_t output_size_;
    
    friend class Yolov8;
};
