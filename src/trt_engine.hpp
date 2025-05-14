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

#include "cuda_utils.hpp"

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
    nvinfer1::IRuntime* m_runtime;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_ctx;

    std::string m_input_name;
    std::string m_output_name;

    size_t m_input_size; 
    size_t m_output_size;
    
    friend class Yolov8;
};
