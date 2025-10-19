#pragma once

#include <NvInfer.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include "file_io.hpp"

#include "cuda_runtime.h"

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

// Cuda error check macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
