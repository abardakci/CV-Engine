#pragma once

#include <NvInfer.h>
#include <string>

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
