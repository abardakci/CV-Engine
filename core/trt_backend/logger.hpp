#pragma once

#include <iostream>
#include <string>
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger 
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override 
    {
        if (severity <= ILogger::Severity::kERROR)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

extern Logger gLogger;
